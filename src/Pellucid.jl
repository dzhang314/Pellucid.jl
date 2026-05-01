module Pellucid

using Base.Iterators: partition
using BFloat16s: BFloat16
using JSON: parsefile
using Libdl: dlext, dlopen, dlsym
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
    added_tokens::AbstractVector{AddedToken},
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


struct NFCNormalizer <: AbstractNormalizer end


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


struct ByteLevelPretokenizer <: AbstractPretokenizer end


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


export AbstractTokenizerModel, vocabulary,
    BPETokenizerModel, construct_tokenizer_model


abstract type AbstractTokenizerModel end


(::AbstractTokenizerModel)(token_id::Integer) = Int[Int(token_id)]


struct BPETokenizerModel <: AbstractTokenizerModel
    vocabulary::Dict{String,Int}
    merges::Dict{Tuple{Int,Int},Tuple{Int,Int}}
end


@inline vocabulary(model::BPETokenizerModel) = model.vocabulary


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


####################################################################### DECODERS


export AbstractDecoder, ByteLevelDecoder, construct_decoder


abstract type AbstractDecoder end


struct ByteLevelDecoder <: AbstractDecoder end


@inline function (::ByteLevelDecoder)(c::Char)
    k = codepoint(c)
    if (0x21 <= k <= 0x7E) | (0xA1 <= k <= 0xAC) | (0xAE <= k <= 0xFF)
        return k % UInt8
    elseif 0x0100 <= k <= 0x0120
        return (k - 0x0100) % UInt8
    elseif 0x0121 <= k <= 0x0142
        return (k - 0x00A2) % UInt8
    elseif k == 0x0143
        return (k - 0x0096) % UInt8
    end
    return nothing
end


function (decoder::ByteLevelDecoder)(s::AbstractString)
    result = UInt8[]
    for c in s
        b = decoder(c)
        if isnothing(b)
            return Vector{UInt8}(s)
        end
        push!(result, b)
    end
    return result
end


function construct_decoder(decoder_json)
    if decoder_json["type"] == "ByteLevel"
        return ByteLevelDecoder()
    end
    error("Unknown decoder type: $(decoder_json["type"])")
end


##################################################################### TOKENIZERS


export Tokenizer, construct_tokenizer


struct Tokenizer
    added_tokens::AbstractVector{AddedToken}
    normalizer::AbstractNormalizer
    pretokenizer::AbstractPretokenizer
    model::AbstractTokenizerModel
    decoder::AbstractDecoder
    token_bytes::Dict{Int,Vector{UInt8}}
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
    added_tokens = AddedToken.(tokenizer_json["added_tokens"])
    model = construct_tokenizer_model(tokenizer_json["model"])
    decoder = construct_decoder(tokenizer_json["decoder"])
    return Tokenizer(
        added_tokens,
        construct_normalizer(tokenizer_json["normalizer"]),
        construct_pretokenizer(tokenizer_json["pre_tokenizer"]),
        model,
        decoder,
        merge(
            Dict(id => decoder(s) for (s, id) in vocabulary(model)),
            Dict(t.id => decoder(t.content) for t in added_tokens)))
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


######################################################### LINEAR ALGEBRA KERNELS


export linear!


const PELLUCID_LIBRARY = Ref{Ptr{Cvoid}}()
const PELLUCID_MATVEC_BF16 = Ref{Ptr{Cvoid}}()
const PELLUCID_MATMUL_BF16 = Ref{Ptr{Cvoid}}()


function __init__()
    PELLUCID_LIBRARY[] = dlopen(joinpath(@__DIR__, "..",
        "deps", "usr", "lib", "libpellucid.$dlext"))
    PELLUCID_MATVEC_BF16[] = dlsym(PELLUCID_LIBRARY[], :pellucid_matvec_bf16)
    PELLUCID_MATMUL_BF16[] = dlsym(PELLUCID_LIBRARY[], :pellucid_matmul_bf16)
    return nothing
end


function linear!(
    y::StridedVector{BFloat16},
    w::StridedMatrix{BFloat16},
    x::StridedVector{BFloat16},
)
    @assert strides(y) == (1,)
    @assert strides(w) == (1, size(w, 1))
    @assert strides(x) == (1,)
    @assert axes(w, 1) == axes(x, 1)
    @assert axes(w, 2) == axes(y, 1)
    @assert iszero(size(w, 1) % 128)
    ccall(PELLUCID_MATVEC_BF16[], Cvoid,
        (Ptr{BFloat16}, Ptr{BFloat16}, Ptr{BFloat16}, Csize_t, Csize_t),
        y, w, x, length(y), length(x))
    return y
end


function linear!(
    y::StridedMatrix{BFloat16},
    w::StridedMatrix{BFloat16},
    x::StridedMatrix{BFloat16},
)
    @assert strides(y) == (1, size(y, 1))
    @assert strides(w) == (1, size(w, 1))
    @assert strides(x) == (1, size(x, 1))
    @assert axes(w, 1) == axes(x, 1)
    @assert axes(w, 2) == axes(y, 1)
    @assert axes(x, 2) == axes(y, 2)
    @assert iszero(size(w, 1) % 32)
    @assert iszero(size(w, 2) % 8)
    @assert iszero(size(x, 1) % 128) || iseven(size(x, 2))
    batch_size = size(x, 2)
    kernel_batch_size = batch_size - (batch_size % 2)
    if !iszero(kernel_batch_size)
        ccall(PELLUCID_MATMUL_BF16[], Cvoid,
            (Ptr{BFloat16}, Ptr{BFloat16}, Ptr{BFloat16},
                Csize_t, Csize_t, Csize_t),
            y, w, x, size(y, 1), kernel_batch_size, size(x, 1))
    end
    if isodd(batch_size)
        linear!(view(y, :, last(axes(y, 2))), w, view(x, :, last(axes(x, 2))))
    end
    return y
end


############################################################ LAYER NORMALIZATION


export rmsnorm!


function rmsnorm!(
    x::AbstractVector{BFloat16},
    w::AbstractVector{BFloat16},
    epsilon::Float32,
)
    @assert axes(x, 1) == axes(w, 1)
    @inbounds begin
        acc = zero(Float32)
        @simd for i in eachindex(x)
            acc += abs2(Float32(x[i]))
        end
        inv_rms = sqrt(inv(acc / Float32(length(x)) + epsilon))
        @simd ivdep for i in eachindex(x)
            x[i] = BFloat16(Float32(w[i]) * (inv_rms * Float32(x[i])))
        end
    end
    return x
end


function rmsnorm!(
    x::AbstractMatrix{BFloat16},
    w::AbstractVector{BFloat16},
    epsilon::Float32,
)
    @assert axes(x, 1) == axes(w, 1)
    for i in axes(x, 2)
        rmsnorm!(view(x, :, i), w, epsilon)
    end
    return x
end


function rmsnorm!(
    x::AbstractArray{BFloat16,3},
    w::AbstractVector{BFloat16},
    epsilon::Float32,
)
    @assert axes(x, 1) == axes(w, 1)
    for j in axes(x, 3)
        for i in axes(x, 2)
            rmsnorm!(view(x, :, i, j), w, epsilon)
        end
    end
    return x
end


###################################################### ROTARY POSITION EMBEDDING


export rope_angular_velocities, rope!


function rope_angular_velocities(base::T, d_head::Integer) where {T}
    @assert iseven(d_head)
    multiplier = -log2(base) / T(d_head)
    return [exp2(T(2 * i) * multiplier) for i = 0:div(Int(d_head), 2)-1]
end


function rope!(
    x::AbstractVector{BFloat16},
    angular_velocities::AbstractVector{Float32},
    position::Integer,
)
    offset = length(angular_velocities)
    @assert axes(x, 1) == Base.OneTo(2 * offset)
    @assert axes(angular_velocities, 1) == Base.OneTo(offset)
    p = Float32(position)
    @inbounds begin
        @simd ivdep for i = 1:offset
            s, c = sincos(p * angular_velocities[i])
            j = i + offset
            u = Float32(x[i])
            v = Float32(x[j])
            x[i] = BFloat16(c * u - s * v)
            x[j] = BFloat16(s * u + c * v)
        end
    end
    return x
end


function rope!(
    x::AbstractMatrix{BFloat16},
    angular_velocities::AbstractVector{Float32},
    position::Integer,
)
    @inbounds for i in axes(x, 2)
        rope!(view(x, :, i), angular_velocities, position)
    end
    return x
end


function rope!(
    x::AbstractArray{BFloat16,3},
    angular_velocities::AbstractVector{Float32},
    positions::AbstractVector{<:Integer},
)
    @assert axes(x, 3) == axes(positions, 1)
    @inbounds for j in axes(x, 3)
        for i in axes(x, 2)
            rope!(view(x, :, i, j), angular_velocities, positions[j])
        end
    end
    return x
end


########################################################### ACTIVATION FUNCTIONS


export silu, softmax!


@inline silu(x::T) where {T} = x / (one(T) + exp(-x))


function softmax!(x::AbstractVector{T}) where {T}
    if !isempty(x)
        @inbounds begin
            m = maximum(x)
            acc = zero(T)
            @simd for i in eachindex(x)
                y = exp(x[i] - m)
                x[i] = y
                acc += y
            end
            inv_sum = inv(acc)
            @simd ivdep for i in eachindex(x)
                x[i] *= inv_sum
            end
        end
    end
    return x
end


###################################################################### ATTENTION


export causal_attention_prefill!, causal_attention_decode!


@inline grouped_zip(xs, ys) = (
    (x, y)
    for (y, group) in zip(ys, partition(xs, div(length(xs), length(ys))))
    for x in group)


function causal_attention_prefill!(
    z::AbstractArray{BFloat16,3},
    q::AbstractArray{BFloat16,3},
    k::AbstractArray{BFloat16,3},
    v::AbstractArray{BFloat16,3},
    scores::AbstractVector{Float32};
    scale::Float32=sqrt(inv(Float32(size(q, 1)))),
    cache_indices::AbstractVector{<:Integer}=axes(q, 3),
)
    ax_head = axes(q, 1)
    ax_q_heads = axes(q, 2)
    ax_q_tokens = axes(q, 3)
    ax_kv_heads = axes(k, 2)
    ax_kv_tokens = axes(k, 3)
    @assert axes(z) == (ax_head, ax_q_heads, ax_q_tokens)
    @assert axes(q) == (ax_head, ax_q_heads, ax_q_tokens)
    @assert axes(k) == (ax_head, ax_kv_heads, ax_kv_tokens)
    @assert axes(v) == (ax_head, ax_kv_heads, ax_kv_tokens)
    @assert issubset(ax_kv_tokens, axes(scores, 1))
    @assert length(cache_indices) == length(ax_q_tokens)
    @assert issubset(cache_indices, ax_kv_tokens)
    @assert iszero(length(ax_q_heads) % length(ax_kv_heads))
    @inbounds for (t_q, t_kv) in zip(ax_q_tokens, cache_indices)
        for (h_q, h_kv) in grouped_zip(ax_q_heads, ax_kv_heads)
            for s = first(ax_kv_tokens):t_kv
                acc = zero(Float32)
                @simd for i in ax_head
                    acc += Float32(q[i, h_q, t_q]) * Float32(k[i, h_kv, s])
                end
                scores[s] = scale * acc
            end
            softmax!(view(scores, first(ax_kv_tokens):t_kv))
            for i in ax_head
                acc = zero(Float32)
                @simd for s = first(ax_kv_tokens):t_kv
                    acc += scores[s] * Float32(v[i, h_kv, s])
                end
                z[i, h_q, t_q] = BFloat16(acc)
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
    scores::AbstractVector{Float32};
    scale::Float32=sqrt(inv(Float32(size(q, 1)))),
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
    @assert iszero(length(ax_q_heads) % length(ax_kv_heads))
    @inbounds for (h_q, h_kv) in grouped_zip(ax_q_heads, ax_kv_heads)
        for s in ax_tokens
            acc = zero(Float32)
            @simd for i in ax_head
                acc += Float32(q[i, h_q]) * Float32(k[i, h_kv, s])
            end
            scores[s] = scale * acc
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
