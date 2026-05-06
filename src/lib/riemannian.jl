"""
Calculation of the Riemannian natural gradient using the method from "Handling the Positive-Definite Constraint in the Bayesian Learning Rule" (Lin et al., 2020).
"""

"""
    natgrad(o::NaturalDescent{FullNormal,RiemannianManifold}, state, dx)

Calculate the Riemannian Natural Gradient for a Full-Rank Gaussian parameterized by the mean `m`
and precision matrix `S` (where S = Σ⁻¹).

### Mathematical Justification (Lin et al., 2020):
The variational parameter is λ = (μ, S), where S is the precision matrix.
The gradient of the variational objective with respect to the covariance Σ is:
    ∇_Σ = 0.5 * E_q[Σ⁻¹(z - μ) * ∇_zᵀ ℓ] - 0.5 * τ * Σ⁻¹

Using the precision Cholesky factor S = L*Lᵀ, the term Σ⁻¹(z - μ) simplifies exactly to L * ϵ.
Thus, the expected covariance gradient is estimated as:
    S̄ = L * E_q[ϵ * ∇_zᵀ ℓ]
    ∇_Σ = 0.25 * (S̄ + S̄ᵀ) - 0.5 * τ * S

The natural gradient of the precision matrix S is defined as (using the paper's Ĝ notation):
    Ĝ = -2 * ∇_Σ

To perform the update on the Riemannian manifold of positive-definite matrices Sym⁺(d), we use
the second-order geodesic expansion (Eq 9, Lin et al., 2020):
    ∇̃S' = η * Ĝ - (η²/2) * Ĝ * Σ * Ĝ
"""
function natgrad(o::NaturalDescent{FullNormal,RiemannianManifold}, state, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    η, τ = T(o.eta), T(o.tau)
    _, S = state.q
    chol_S = cholesky(S).L
    Σ = inv(S)

    # natgrad m = Σ⋅∂μℓ, ∂μℓ ≈ E_q[∇z logjoint(z)]
    ∇m = Σ * mean(dx)
    # Estimate covariance gradient using ∂Σℓ ≈ 0.5 ⋅ E_q[Σ⁻¹(z - μ) ⋅ ∇zᵀ logjoint(z)] (alternative is 0.5 ⋅ E_q[∇²z logjoint(z)])
    S̄ = chol_S * mean(state.epsilon .* transpose.(dx))
    # Ensure symmetry and add entropy gradient (-0.5 ⋅ Σ⁻¹ = -0.5 ⋅ S) multiplied by temperature τ.
    ∇Σ = T(0.25) * Symmetric(S̄ + S̄') + τ * -T(0.5) * S
    Ĝ = -2∇Σ # Natural gradient of S: -2∂Σℓ
    # η * Ĝ minus the correction because we subtract later:
    ∇S = η * Ĝ - abs2(η) / 2 * Ĝ * Σ * Ĝ

    # undo the learning rate multiplication for ∇̃S because we also do this in update function:
    return ∇m, 1 / η * ∇S
end

"""
    natgrad(o::NaturalDescent{DiagNormal,RiemannianManifold}, state, dx)

Calculate the Riemannian Natural Gradient for a Diagonal Gaussian parameterized by the mean `m`
and precision vector `s` (where s = σ⁻²).

### Mathematical Justification:
The diagonal version of the Riemannian geodesic step uses s = diag(S) and the diagonal of Ĝ.
The standard deviation scale factor is t = √s, allowing the covariance gradient term to be computed as:
    S̄ = t * E_q[ϵ * ∇_z ℓ]
"""
function natgrad(o::NaturalDescent{DiagNormal,RiemannianManifold}, state, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    η, τ = T(o.eta), T(o.tau)
    _, s = state.q
    t = sqrt.(s)
    σ² = 1 ./ s

    ∇m = σ² .* mean(dx)

    S̄ = t .* mean(map(.*, state.epsilon, dx))
    ∇σ² = T(0.5) .* S̄ + τ .* -T(0.5) .* s
    Ĝ = -2∇σ²
    # η * Ĝ minus the correction because we subtract later:
    ∇s = η .* Ĝ - abs2(η) / 2 .* Ĝ .* σ² .* Ĝ

    # undo the learning rate multiplication for ∇̃S because we also do this in update function:
    return ∇m, 1 / η * ∇s
end

"""
    update(o::NaturalDescent{FullNormal,RiemannianManifold}, m, S, ∇̃μ, ∇̃S)

Perform the parameter update step on the Full-Rank Gaussian Riemannian manifold.
"""
function update(o::NaturalDescent{FullNormal,RiemannianManifold}, m::AbstractArray{T}, S::Symmetric, ∇̃μ, ∇̃S) where T<:Real
    η = T(o.eta)
    return m - η * ∇̃μ, Symmetric(S - η * ∇̃S)
end

"""
    update(o::NaturalDescent{DiagNormal,RiemannianManifold}, m, s, ∇̃μ, ∇̃s)

Perform the parameter update step on the Diagonal Gaussian Riemannian manifold.
"""
function update(o::NaturalDescent{DiagNormal,RiemannianManifold}, m, s::AbstractArray{T}, ∇̃μ, ∇̃s) where T<:Real
    η = T(o.eta)
    return m - η .* ∇̃μ, s - η .* ∇̃s
end

"""
    sample(::NaturalDescent, state, i)

Sample from the variational distribution on the Riemannian manifold.
"""
function sample(::NaturalDescent{FullNormal,RiemannianManifold}, state, i::Int)
    m, S = state.q
    T = cholesky(S).L
    return inv(T') * state.epsilon[i] + m
end

function sample(::NaturalDescent{DiagNormal,RiemannianManifold}, state, i::Int)
    m, s = state.q # s is the diagonal of the precision matrix S
    σ = @. sqrt(1 / s) # diagonal of covariance matrix
    return σ .* state.epsilon[i] + m
end
