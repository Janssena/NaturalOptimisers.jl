abstract type Parameterisation end

struct MeanVar <: Parameterisation end # μ, Σ or σ²
struct MeanPrec <: Parameterisation end # μ, S or s
struct MeanSqrt <: Parameterisation end # μ, L or σ

# Uses MeanPrec
struct RiemannianManifold <: AbstractManifold end
# Uses MeanSqrt
struct LieGroupManifold <: AbstractManifold end
# Uses MeanSqrt
struct EuclidianManifold <: AbstractManifold end

initq(o::NaturalDescent, x::Distribution; kwargs...) = nothing # use the q in the ps object
initq(o::NaturalDescent{Q,RiemannianManifold}, x; kwargs...) where Q =
    initq(Q, MeanPrec(), x; kwargs...)

initq(o::NaturalDescent{Q,M}, x; kwargs...) where {Q,M<:Union{LieGroupManifold,EuclidianManifold}} =
    initq(Q, MeanSqrt(), x; kwargs...)


initq(::Type{FullNormal}, params::Parameterisation, x::AbstractArray; kwargs...) =
    initq(q, params, reshape(x, length(x)); kwargs...)

initq(::Type{FullNormal}, ::MeanPrec, x::AbstractVector; scale::Real) =
    zero(x), Symmetric(collect(Diagonal(1/scale .* one.(x)))) # m, S

initq(::Type{DiagNormal}, ::MeanPrec, x::AbstractArray; scale::Real) =
    zero(x), 1/scale .* one.(x) # m, s

initq(::Type{FullNormal}, ::MeanSqrt, x::AbstractVector; scale::Real) =
    zero(x), LowerTriangular(collect(Diagonal(scale .* one.(x)))) # m, L

initq(::Type{DiagNormal}, ::MeanSqrt, x::AbstractArray; scale::Real) =
    zero(x), scale .* one.(x) # m, σ