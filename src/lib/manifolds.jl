"""
    Parameterisation

How the second-order state of a Gaussian `q` is stored. Each [`AbstractManifold`](@ref)
picks one, which determines what `initq` seeds and what `natgrad`/`update` operate on:

- [`MeanVar`](@ref)  — mean and (co)variance `Σ` / `σ²`.
- [`MeanPrec`](@ref) — mean and precision `S = Σ⁻¹` / `s = σ⁻²`.
- [`MeanSqrt`](@ref) — mean and a square-root scale `L` (with `Σ = LLᵀ`) / `σ`.
"""
abstract type Parameterisation end

"Mean–(co)variance parameterisation `(μ, Σ)` / `(μ, σ²)`. See [`Parameterisation`](@ref)."
struct MeanVar <: Parameterisation end # μ, Σ or σ²
"Mean–precision parameterisation `(μ, S = Σ⁻¹)` / `(μ, s = σ⁻²)`, used by [`RiemannianManifold`](@ref)."
struct MeanPrec <: Parameterisation end # μ, S or s
"Mean–square-root-scale parameterisation `(μ, L)` (`Σ = LLᵀ`) / `(μ, σ)`, used by [`LieGroupManifold`](@ref) and [`EuclidianManifold`](@ref)."
struct MeanSqrt <: Parameterisation end # μ, L or σ

"""
    RiemannianManifold()

Scale geometry for [`NaturalDescent`](@ref) that tracks the **precision**
`S = Σ⁻¹` ([`MeanPrec`](@ref)) and updates it along geodesics of the
positive-definite cone Sym⁺(d) with a second-order correction that keeps `S ≻ 0`.
From Lin et al. (2020), *Handling the Positive-Definite Constraint in the
Bayesian Learning Rule*. See [`natgrad`](@ref)/[`update`](@ref) in `riemannian.jl`.
"""
struct RiemannianManifold <: AbstractManifold end
"""
    LieGroupManifold()

Default scale geometry for [`NaturalDescent`](@ref). Tracks a **Cholesky scale**
`L` with `Σ = LLᵀ` ([`MeanSqrt`](@ref)) and updates it by the exponential map of
the affine Lie group, `L ← L·exp(-ηU)`, which preserves positive-definiteness
exactly. From Kıral et al. (2023), *The Lie-Group Bayesian Learning Rule*. See
[`natgrad`](@ref)/[`update`](@ref) in `lie_groups.jl`.
"""
struct LieGroupManifold <: AbstractManifold end
"""
    EuclidianManifold()

Scale geometry for [`NaturalDescent`](@ref) using the standard Euclidean/Cholesky
natural gradient under the Fisher information metric ([`MeanSqrt`](@ref)): a
Cholesky factor `L` for `FullNormal`, and an unconstrained `ϕ` with
`σ = softplus(ϕ)` for `DiagNormal`. In the lineage of Khan et al. (2018) /
the Bayesian Learning Rule. See [`natgrad`](@ref)/[`update`](@ref) in `euclidian.jl`.
"""
struct EuclidianManifold <: AbstractManifold end

"""
    initq(o::NaturalDescent, x; scale)

Initialise the variational parameters `q` of [`NaturalDescent`](@ref) for a
parameter `x`, in the parameterisation dictated by the rule's manifold
([`MeanPrec`](@ref) for `RiemannianManifold`, [`MeanSqrt`](@ref) otherwise) and
its family (`DiagNormal` vs `FullNormal`). Returns a tuple of the zero mean and
an isotropic scale set from `scale` (the initial standard deviation): the
covariance is `scale²·I`, stored as a precision `1/scale²`, a Cholesky factor
`scale·I`, or a diagonal `scale`, as appropriate.

When `x` is already a `Distribution`, returns `nothing` — the `q` carried in the
model object is used directly.
"""
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