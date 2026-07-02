"""
Improved Variational Online Newton (IVON) from "Variational Learning is Effective
for Large Deep Networks" (Shen et al., 2024).

IVON is a mean-field variational method that learns a diagonal Gaussian
`q(θ) = N(θ | m, diag(σ²))` over the network weights with `σ² = 1/(λ(h + δ))`,
where `h` is a running estimate of the (positive) diagonal Hessian, `δ > 0` is a
weight-decay / prior precision (prior `N(0, (λδ)⁻¹ I)`), and `λ` is the
variational dataset scaling (`= N` for pure Bayesian inference). It targets the
objective `L(q) = λ Eq[ℓ̄(θ)] + KL(q ‖ p)`.

The update closely resembles Adam (Alg. 1):

    ĝ  ← ∇ℓ̄(θ),   θ ∼ q                          # gradient at a sampled weight
    ĥ  ← ĝ ⊙ (θ - m)/σ²  =  ĝ ⊙ ϵ/σ               # reparameterised Hessian estimate
    g  ← β₁ g + (1 - β₁) ĝ                         # gradient momentum
    h  ← β₂ h + (1 - β₂) ĥ + ½(1 - β₂)² (h - ĥ)²/(h + δ)   # Hessian EMA + positivity correction
    ḡ  ← g / (1 - β₁ᵗ)                             # bias correction (gradient only)
    m  ← m - αₜ (ḡ + δ m)/(h + δ)                  # Newton-like mean update with prior
    σ  ← 1/√(λ(h + δ))

This is the diagonal special case of [`EVON`](@ref) with an identity rotation,
but IVON debiases the gradient momentum (Alg. 1, line 6) whereas EVON's Alg. 2
does not. IVON operates element-wise and so supports parameters of any shape.
"""

"""
    IVON(eta, beta=(0.9, 0.99995); delta=1, lambda=1, init_scale=0.01, clip=Inf, rescale=false)

Construct the IVON optimisation rule.

# Arguments
- `eta`: mean-update learning rate `αₜ` (Alg. 1, line 7).
- `beta = (β₁, β₂)`: EMA decay rates for the gradient momentum `g` (line 4) and
  the Hessian estimate `h` (line 5).

# Keywords
- `delta`: weight-decay / prior precision `δ > 0`; the prior is `N(0, (λδ)⁻¹ I)`.
- `lambda`: variational dataset scaling `λ` in `L(q) = λ Eq[ℓ̄] + KL` (typically
  the number of data points `N`).
- `init_scale`: initial Hessian value `h₀ > 0` (the paper uses `h₀ ≈ 0.01`).
- `clip`: element-wise clip bound `ξ` on the preconditioned update (line 7);
  defaults to `Inf` (disabled).
- `rescale`: if `true`, rescale the learning rate by `(h₀ + δ)` so the first
  steps take a step-size close to `αₜ` (Alg. 1, optional). Omitted when clipping.
- `init_mean`: if `true`, warm-start the variational mean `m` at the parameter's current
  value (Alg. 1 initialises `m ← NN-weights`, e.g. for fine-tuning a checkpoint); otherwise
  `m = 0` to match the rest of this package's convention.
"""
struct IVON{T} <: AbstractNaturalRule
    eta::T              # αₜ, mean learning rate
    beta::NTuple{2,T}   # (β₁ momentum, β₂ Hessian EMA)
    delta::T            # δ, weight decay / prior precision
    lambda::T           # λ, variational dataset scaling
    init_scale::T       # h₀, initial Hessian
    clip::T             # ξ, element-wise clip bound (Inf = disabled)
    init_mean::Bool     # warm-start m at the parameter (vs zero)
end

function IVON(eta, beta=(0.9, 0.99995); delta=1, lambda=1, init_scale=0.01, clip=Inf, rescale=false, init_mean=false)
    (delta <= 0) && throw(ErrorException("`delta` (prior precision) must be positive."))
    (lambda <= 0) && throw(ErrorException("`lambda` (dataset scaling) must be positive."))
    (init_scale <= 0) && throw(ErrorException("`init_scale` (Hessian init h₀) must be positive."))

    T = eltype(eta)
    α = rescale ? (T(init_scale) + T(delta)) * T(eta) : T(eta)
    return IVON{T}(α, T.(beta), T(delta), T(lambda), T(init_scale), T(clip), init_mean)
end

function Optimisers.init(o::IVON, x::AbstractArray{T}) where T<:Real
    β₁ = o.beta[1]
    m = o.init_mean ? copy(x) : zero(x)               # warm-start at the parameter, or start at zero
    return (
        q=(m, fill(T(o.init_scale), size(x))),        # mean m, Hessian h
        momentum=(zero(x), β₁),                       # gradient momentum g, running β₁ᵗ
        epsilon=[zero(x)]                             # standard-normal noise samples ϵ
    )
end

Optimisers.apply!(o::IVON, state, x::AbstractArray{T}, dx::AbstractArray{T}) where T<:Real =
    Optimisers.apply!(o, state, x, [dx])

"""
    Optimisers.apply!(o::IVON, state, x, dx)

Perform one IVON step (Alg. 1). `dx` is a vector of per-sample loss gradients
`ĝ = ∇ℓ̄(θ)` evaluated at weights `θ` drawn from the current posterior (see
[`sample`](@ref)). As elsewhere in this package, the model parameter `x` is left
unchanged; the variational state lives in `state` and weights are materialised
via `sample`.
"""
function Optimisers.apply!(o::IVON, state, ::AbstractArray{T}, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    β₁, β₂ = o.beta
    δ, λ, α = o.delta, o.lambda, o.eta
    m, h = state.q
    g, β₁ᵗ = state.momentum
    ϵs = state.epsilon

    σ = @. 1 / sqrt(λ * (h + δ))

    # Reparameterised Hessian estimator ĥ = ĝ ⊙ (θ - m)/σ² = ĝ ⊙ ϵ/σ (line 3),
    # and gradient momentum (line 4), averaged over the posterior samples.
    ĝ = mean(dx)
    ĥ = mean(map((g_s, ϵ) -> g_s .* ϵ ./ σ, dx, ϵs))

    g′ = @. β₁ * g + (1 - β₁) * ĝ
    h′ = @. β₂ * h + (1 - β₂) * ĥ + (1 - β₂)^2 / 2 * (h - ĥ)^2 / (h + δ)

    # Bias-corrected momentum (line 6) and Newton-like mean update with prior (line 7).
    ḡ = g′ ./ (1 - β₁ᵗ)
    Δ = _clip((ḡ .+ δ .* m) ./ (h′ .+ δ), o.clip)
    m′ = m .- α .* Δ

    state′ = (
        q=(m′, h′),
        momentum=(g′, β₁ᵗ * β₁),
        epsilon=ϵs
    )

    return state′, nothing
end

"""
    sample(o::IVON, state, i)

Draw a weight vector from the diagonal posterior using the `i`-th stored
standard-normal noise sample `ϵ`:  `θ = m + σ ⊙ ϵ`,  `σ = 1/√(λ(h + δ))`.
"""
function sample(o::IVON, state, i::Int)
    m, h = state.q
    σ = @. 1 / sqrt(o.lambda * (h + o.delta))
    return @. m + σ * state.epsilon[i]
end
