module Pellucid

using BFloat16s: BFloat16
using JSON: parsefile
using SafeTensors: deserialize
using Unicode: normalize

################################################################### ADDED TOKENS


export AddedToken, Pretoken, split_added_tokens


struct AddedToken
    id::Int
    content::String
    special::Bool
end


function AddedToken(token_json)
    @assert token_json["id"] isa Integer
    @assert token_json["content"] isa AbstractString
    @assert token_json["single_word"] === false
    @assert token_json["lstrip"] === false
    @assert token_json["rstrip"] === false
    @assert token_json["normalized"] === false
    @assert token_json["special"] isa Bool
    return AddedToken(
        Int(token_json["id"]),
        String(token_json["content"]),
        token_json["special"])
end


const Pretoken = Union{Int,String,SubString{String}}


function push_split_added_tokens!(
    result::AbstractVector{Pretoken},
    s::Union{String,SubString{String}},
    added_tokens::AbstractVector{AddedToken}
)
    if isempty(s)
        return result
    end
    # TODO: This algorithm is only correct when no added token contains
    # another, and no prefix of one added token is a suffix of another.
    for added_token in added_tokens
        m = findfirst(added_token.content, s)
        if !isnothing(m)
            push_split_added_tokens!(result,
                SubString(s, firstindex(s), prevind(s, first(m))),
                added_tokens)
            push!(result, added_token.id)
            push_split_added_tokens!(result,
                SubString(s, nextind(s, last(m)), lastindex(s)),
                added_tokens)
            return result
        end
    end
    push!(result, s)
    return result
end


function split_added_tokens(
    s::AbstractString,
    added_tokens::AbstractVector{AddedToken},
)
    if !(s isa Union{String,SubString{String}})
        s = String(s)
    end
    result = Pretoken[]
    push_split_added_tokens!(result, s, added_tokens)
    return result
end


#################################################################### NORMALIZERS


export AbstractNormalizer, NFCNormalizer, construct_normalizer


abstract type AbstractNormalizer end


(::AbstractNormalizer)(token_id::Integer) = Pretoken[Int(token_id)]


struct NFCNormalizer <: AbstractNormalizer
end


(::NFCNormalizer)(s::AbstractString) = Pretoken[normalize(s, :NFC)]


function construct_normalizer(normalizer_json)
    if normalizer_json["type"] == "NFC"
        return NFCNormalizer()
    end
    error("Unknown normalizer type: $(normalizer_json["type"])")
end


################################################################## PRETOKENIZERS


export AbstractPretokenizer,
    SplitPretokenizer, ByteLevelPretokenizer, SequencePretokenizer,
    construct_pretokenizer


abstract type AbstractPretokenizer end


(::AbstractPretokenizer)(token_id::Integer) = Pretoken[Int(token_id)]


struct SplitPretokenizer <: AbstractPretokenizer
    regex::Regex
end


function (pretokenizer::SplitPretokenizer)(s::AbstractString)
    if !(s isa Union{String,SubString{String}})
        s = String(s)
    end
    result = Pretoken[]
    i = firstindex(s)
    for m in eachmatch(pretokenizer.regex, s)
        if i < m.offset
            push!(result, SubString(s, i, prevind(s, m.offset)))
        end
        if !isempty(m.match)
            push!(result, m.match)
        end
        i = m.offset + ncodeunits(m.match)
    end
    if i <= lastindex(s)
        push!(result, SubString(s, i, lastindex(s)))
    end
    return result
end


struct ByteLevelPretokenizer <: AbstractPretokenizer
end


@inline (::ByteLevelPretokenizer)(b::UInt8) =
    (b <= 0x20) ? Char(UInt16(b) + 0x0100) :
    (0x7F <= b <= 0xA0) ? Char(UInt16(b) + 0x00A2) :
    (b == 0xAD) ? Char(UInt16(b) + 0x0096) : Char(b)


(pretokenizer::ByteLevelPretokenizer)(s::AbstractString) =
    Pretoken[String(pretokenizer.(codeunits(s)))]


struct SequencePretokenizer <: AbstractPretokenizer
    pretokenizers::Vector{AbstractPretokenizer}
end


function (pretokenizer::SequencePretokenizer)(s::AbstractString)
    if !(s isa Union{String,SubString{String}})
        s = String(s)
    end
    result = Pretoken[s]
    for p in pretokenizer.pretokenizers
        result = mapreduce(p, vcat, result; init=Pretoken[])
    end
    return result
end


function construct_pretokenizer(pretokenizer_json)
    if pretokenizer_json["type"] == "Split"
        @assert pretokenizer_json["behavior"] == "Isolated"
        @assert pretokenizer_json["invert"] === false
        return SplitPretokenizer(Regex(pretokenizer_json["pattern"]["Regex"]))
    elseif pretokenizer_json["type"] == "ByteLevel"
        @assert pretokenizer_json["add_prefix_space"] === false
        @assert pretokenizer_json["use_regex"] === false
        return ByteLevelPretokenizer()
    elseif pretokenizer_json["type"] == "Sequence"
        return SequencePretokenizer([
            construct_pretokenizer(p)
            for p in pretokenizer_json["pretokenizers"]])
    end
    error("Unknown pretokenizer type: $(pretokenizer_json["type"])")
end


############################################################### TOKENIZER MODELS


export AbstractTokenizerModel, BPETokenizerModel, construct_tokenizer_model


abstract type AbstractTokenizerModel end


(::AbstractTokenizerModel)(token_id::Integer) = Int[Int(token_id)]


struct BPETokenizerModel <: AbstractTokenizerModel
    vocabulary::Dict{String,Int}
    merges::Dict{Tuple{Int,Int},Tuple{Int,Int}}
end


function (model::BPETokenizerModel)(s::AbstractString)
    result = Int[model.vocabulary[string(c)] for c in s]
    while true
        best_merge = ((typemax(Int), 0), 0)
        i = firstindex(result)
        while i < lastindex(result)
            i_next = nextind(result, i)
            key = (result[i], result[i_next])
            if haskey(model.merges, key)
                best_merge = min(best_merge, (model.merges[key], i))
            end
            i = i_next
        end
        (_, merged), index = best_merge
        if iszero(index)
            return result
        end
        result[index] = merged
        deleteat!(result, index + 1)
    end
end


function construct_tokenizer_model(model_json)
    if model_json["type"] == "BPE"
        @assert isnothing(model_json["dropout"]) ||
                iszero(model_json["dropout"])
        @assert isnothing(model_json["unk_token"])
        @assert isnothing(model_json["continuing_subword_prefix"]) ||
                isempty(model_json["continuing_subword_prefix"])
        @assert isnothing(model_json["end_of_word_suffix"]) ||
                isempty(model_json["end_of_word_suffix"])
        @assert model_json["byte_fallback"] === false
        @assert model_json["ignore_merges"] === false
        vocabulary = Dict{String,Int}(model_json["vocab"])
        merges = Dict{Tuple{Int,Int},Tuple{Int,Int}}()
        for (i, (v, w)) in enumerate(model_json["merges"])
            merges[(vocabulary[v], vocabulary[w])] = (i, vocabulary[v*w])
        end
        return BPETokenizerModel(vocabulary, merges)
    end
    error("Unknown tokenizer model type: $(model_json["type"])")
end


##################################################################### TOKENIZERS


export Tokenizer, construct_tokenizer

struct Tokenizer
    added_tokens::AbstractVector{AddedToken}
    normalizer::AbstractNormalizer
    pretokenizer::AbstractPretokenizer
    model::AbstractTokenizerModel
end


function (tokenizer::Tokenizer)(s::AbstractString)
    pieces = split_added_tokens(s, tokenizer.added_tokens)
    normalized = mapreduce(tokenizer.normalizer, vcat, pieces;
        init=Pretoken[])
    pretokens = mapreduce(tokenizer.pretokenizer, vcat, normalized;
        init=Pretoken[])
    return mapreduce(tokenizer.model, vcat, pretokens; init=Int[])
end


function construct_tokenizer(tokenizer_json)
    @assert isnothing(tokenizer_json["truncation"])
    @assert isnothing(tokenizer_json["padding"])
    post_processor_json = tokenizer_json["post_processor"]
    if !isnothing(post_processor_json)
        @assert post_processor_json["type"] == "ByteLevel"
    end
    decoder_json = tokenizer_json["decoder"]
    @assert !isnothing(decoder_json)
    @assert decoder_json["type"] == "ByteLevel"
    return Tokenizer(
        AddedToken.(tokenizer_json["added_tokens"]),
        construct_normalizer(tokenizer_json["normalizer"]),
        construct_pretokenizer(tokenizer_json["pre_tokenizer"]),
        construct_tokenizer_model(tokenizer_json["model"]))
end


################################################################## MODEL LOADING


export load_safetensors_model


function load_safetensors_model(model_dir::AbstractString)
    result = Dict{String,AbstractArray}()
    index_path = joinpath(model_dir, "model.safetensors.index.json")
    if isfile(index_path)
        index_json = parsefile(index_path)
        shards = unique(values(index_json["weight_map"]))
        expected_names = Dict(shard => String[] for shard in shards)
        for (name, shard) in index_json["weight_map"]
            push!(expected_names[shard], name)
        end
        for shard in shards
            found_names = String[]
            for (name, tensor) in deserialize(joinpath(model_dir, shard))
                @assert !haskey(result, name)
                @assert parent(tensor) isa PermutedDimsArray
                result[name] = parent(parent(tensor))
                push!(found_names, name)
            end
            @assert issetequal(found_names, expected_names[shard])
        end
    else
        model_path = joinpath(model_dir, "model.safetensors")
        @assert isfile(model_path)
        for (name, tensor) in deserialize(model_path)
            @assert !haskey(result, name)
            @assert parent(tensor) isa PermutedDimsArray
            result[name] = parent(parent(tensor))
        end
    end
    return result
end


########################################################### ACTIVATION FUNCTIONS


export silu, softmax!


@inline silu(x::T) where {T} = x / (one(T) + exp(-x))


function softmax!(x::AbstractVector{T}) where {T}
    if !isempty(x)
        m = maximum(x)
        s = zero(T)
        @inbounds begin
            @simd for i in eachindex(x)
                y = exp(x[i] - m)
                x[i] = y
                s += y
            end
        end
        x .*= inv(s)
    end
    return x
end


###################################################################### ATTENTION


export causal_attention_prefill!, causal_attention_decode!


function causal_attention_prefill!(
    z::AbstractArray{BFloat16,3},
    q::AbstractArray{BFloat16,3},
    k::AbstractArray{BFloat16,3},
    v::AbstractArray{BFloat16,3},
    scores::AbstractVector{Float32},
)
    ax_head = axes(q, 1)
    ax_q_heads = axes(q, 2)
    ax_tokens = axes(q, 3)
    ax_kv_heads = axes(k, 2)
    @assert axes(z) == (ax_head, ax_q_heads, ax_tokens)
    @assert axes(q) == (ax_head, ax_q_heads, ax_tokens)
    @assert axes(k) == (ax_head, ax_kv_heads, ax_tokens)
    @assert axes(v) == (ax_head, ax_kv_heads, ax_tokens)
    @assert issubset(ax_tokens, axes(scores, 1))
    num_q_heads = length(ax_q_heads)
    num_kv_heads = length(ax_kv_heads)
    @assert iszero(num_q_heads % num_kv_heads)
    g = div(num_q_heads, num_kv_heads)
    inv_sqrt_d_head = sqrt(inv(Float32(length(ax_head))))
    @inbounds for t in ax_tokens
        for (n_q, h_q) in enumerate(ax_q_heads)
            h_kv = ax_kv_heads[div(n_q - 1, g)+1]
            for s = first(ax_tokens):t
                acc = zero(Float32)
                @simd for i in ax_head
                    acc += Float32(q[i, h_q, t]) * Float32(k[i, h_kv, s])
                end
                scores[s] = inv_sqrt_d_head * acc
            end
            softmax!(view(scores, first(ax_tokens):t))
            for i in ax_head
                acc = zero(Float32)
                @simd for s = first(ax_tokens):t
                    acc += scores[s] * Float32(v[i, h_kv, s])
                end
                z[i, h_q, t] = BFloat16(acc)
            end
        end
    end
    return z
end


function causal_attention_decode!(
    z::AbstractMatrix{BFloat16},
    q::AbstractMatrix{BFloat16},
    k::AbstractArray{BFloat16,3},
    v::AbstractArray{BFloat16,3},
    scores::AbstractVector{Float32},
)
    ax_head = axes(q, 1)
    ax_q_heads = axes(q, 2)
    ax_kv_heads = axes(k, 2)
    ax_tokens = axes(k, 3)
    @assert axes(z) == (ax_head, ax_q_heads)
    @assert axes(q) == (ax_head, ax_q_heads)
    @assert axes(k) == (ax_head, ax_kv_heads, ax_tokens)
    @assert axes(v) == (ax_head, ax_kv_heads, ax_tokens)
    @assert issubset(ax_tokens, axes(scores, 1))
    num_q_heads = length(ax_q_heads)
    num_kv_heads = length(ax_kv_heads)
    @assert iszero(num_q_heads % num_kv_heads)
    g = div(num_q_heads, num_kv_heads)
    inv_sqrt_d_head = sqrt(inv(Float32(length(ax_head))))
    @inbounds for (n_q, h_q) in enumerate(ax_q_heads)
        h_kv = ax_kv_heads[div(n_q - 1, g)+1]
        for s in ax_tokens
            acc = zero(Float32)
            @simd for i in ax_head
                acc += Float32(q[i, h_q]) * Float32(k[i, h_kv, s])
            end
            scores[s] = inv_sqrt_d_head * acc
        end
        softmax!(view(scores, ax_tokens))
        for i in ax_head
            acc = zero(Float32)
            @simd for s in ax_tokens
                acc += scores[s] * Float32(v[i, h_kv, s])
            end
            z[i, h_q] = BFloat16(acc)
        end
    end
    return z
end


####################################################################### SAMPLING


export sample_logits


function sample_logits(
    logits::AbstractVector{BFloat16},
    temperature::Real;
    top_k::Union{Nothing,Integer}=nothing,
    top_p::Union{Nothing,Real}=nothing,
)
    @assert !isempty(logits)
    @assert isnothing(top_k) || (top_k > 0)
    @assert isnothing(top_p) || (top_p > 0)
    k = isnothing(top_k) ? length(logits) : min(Int(top_k), length(logits))

    @inbounds begin
        probabilities = similar(logits, Float32)
        probabilities .= Float32.(logits) ./ Float32(temperature)
        softmax!(probabilities)

        top_indices = partialsortperm(probabilities, 1:k, rev=true)
        top_probabilities = probabilities[top_indices]
        top_probabilities ./= sum(top_probabilities)

        total_probability = zero(Float32)
        n = k
        for i = 1:k
            total_probability += top_probabilities[i]
            if (!isnothing(top_p)) && (total_probability >= top_p)
                n = i
                break
            end
        end

        acc = zero(Float32)
        u = rand(Float32) * total_probability
        for i = 1:n
            acc += top_probabilities[i]
            if acc > u
                return top_indices[i]
            end
        end
        return top_indices[n]
    end
end


################################################################################

end # module Pellucid
