module Pellucid

using Unicode: normalize

#################################################################### NORMALIZERS


export AbstractNormalizer, NFCNormalizer, construct_normalizer


const Pretoken = Union{Int,String,SubString{String}}


abstract type AbstractNormalizer end


(::AbstractNormalizer)(token_id::Integer) = Pretoken[Int(token_id)]


struct NFCNormalizer <: AbstractNormalizer
end


(::NFCNormalizer)(s::AbstractString) = Pretoken[normalize(s, :NFC)]


function construct_normalizer(normalizer_json)
    if normalizer_json.type == "NFC"
        return NFCNormalizer()
    end
    error("Unknown normalizer type: $(normalizer_json.type)")
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
    if pretokenizer_json.type == "Split"
        @assert pretokenizer_json.behavior == "Isolated"
        @assert pretokenizer_json.invert === false
        return SplitPretokenizer(Regex(pretokenizer_json.pattern.Regex))
    elseif pretokenizer_json.type == "ByteLevel"
        @assert pretokenizer_json.add_prefix_space === false
        @assert pretokenizer_json.use_regex === false
        return ByteLevelPretokenizer()
    elseif pretokenizer_json.type == "Sequence"
        return SequencePretokenizer([
            construct_pretokenizer(p)
            for p in pretokenizer_json.pretokenizers])
    end
    error("Unknown pretokenizer type: $(pretokenizer_json.type)")
end


################################################################################

end # module Pellucid
