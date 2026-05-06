"""
Calculation of the natural gradient under the Euclidean parameterisation.
"""

"""
    natgrad(o::NaturalDescent{FullNormal,EuclidianManifold}, state, dx)

Calculate the Euclidean Natural Gradient for a Full-Rank Gaussian parameterized by the mean `m` 
and Cholesky factor `L` (where Σ = L * Lᵀ).

### Mathematical Justification:
The variational distribution is parameterized as θ = Lϵ + m. 
The natural gradient with respect to Σ under the Fisher Information Metric (FIM) is:
    ∇̃Σ = 2 * Σ * ∇_Σ * Σ

Using the chain rule relation for the Cholesky factor Σ = L * Lᵀ:
    ∇_L = 2 * ∇_Σ * L  =>  Lᵀ * ∇_L = 2 * Lᵀ * ∇_Σ * L

Projecting this onto the lower-triangular manifold with positive diagonals yields:
    ∇̃L = L * (LowerTriangular(Lᵀ * ∇_L) - 0.5 * Diagonal(Lᵀ * ∇_L))

This matches the standard formulation of the Cholesky natural gradient.
"""
function natgrad(o::NaturalDescent{FullNormal,EuclidianManifold}, state, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    τ = T(o.tau)
    _, L = state.q

    L⁻¹ = inv(L)
    L⁻ᵀ = transpose(L⁻¹)
    S = Symmetric(L⁻ᵀ * L⁻¹)

    ∇̃m = Symmetric(L * L') * mean(dx)

    ∇Σ = T(0.5) * L⁻ᵀ * mean(state.epsilon .* transpose.(dx)) + τ * -T(0.5) * S
    ∇L = 2∇Σ * L
    Lᵀ∇L = transpose(L) * ∇L
    ∇̃L = L * (LowerTriangular(Lᵀ∇L) - T(0.5) * Diagonal(Lᵀ∇L))

    return ∇̃m, ∇̃L
end

"""
    natgrad(o::NaturalDescent{DiagNormal,EuclidianManifold}, state, dx)

Calculate the Euclidean Natural Gradient for a Diagonal Gaussian parameterized by the mean `m`
and unconstrained scale parameter `ϕ` (where σ = softplus(ϕ)).

### Mathematical Justification:
With σ = softplus(ϕ), we have dσ/dϕ = logistic(ϕ).
The induced Fisher Information Matrix with respect to the unconstrained parameter ϕ is:
    F_ϕ = (2 / σ²) * logistic(ϕ)²

By the chain rule, the Euclidean gradient of the variational objective with respect to ϕ is:
    ∇_ϕ = logistic(ϕ) * (E[∇_z ℓ * ϵ] - τ / σ)

The natural gradient is:
    ∇̃ϕ = F_ϕ⁻¹ * ∇_ϕ = [σ / (2 * logistic(ϕ))] * (σ * E[∇_z ℓ * ϵ] - τ)
"""
function natgrad(o::NaturalDescent{DiagNormal,EuclidianManifold}, state, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    τ = T(o.tau)
    _, ϕ = state.q
    σ = softplus.(ϕ)

    # Mean natural gradient: Σ * E[∇zℓ]
    ∇̃m = abs2.(σ) .* mean(dx)

    # ∇σ is the raw scale gradient: σ * E[∇ℓ * ϵ] - τ
    ∇σ = σ .* mean(map(.*, dx, state.epsilon)) .- τ

    # Apply the change of variables factor for the Euclidean Fisher metric
    ∇̃ϕ = σ ./ (2 .* logistic.(ϕ)) .* ∇σ

    return ∇̃m, ∇̃ϕ
end

"""
    update(o::NaturalDescent{FullNormal,EuclidianManifold}, m, L, ∇̃m, ∇̃L)

Perform the parameter update step on the Full-Rank Gaussian Euclidean manifold.
"""
function update(o::NaturalDescent{FullNormal,EuclidianManifold}, m::AbstractArray{T}, L::LowerTriangular, ∇̃m, ∇̃L) where {T<:Real}
    η = T(o.eta)
    return m - η * ∇̃m, LowerTriangular(L - η * ∇̃L)
end

"""
    update(o::NaturalDescent{DiagNormal,EuclidianManifold}, m, ϕ, ∇̃m, ∇̃ϕ)

Perform the parameter update step on the Diagonal Gaussian Euclidean manifold.
"""
function update(o::NaturalDescent{DiagNormal,EuclidianManifold}, m, ϕ::AbstractArray{T}, ∇̃m, ∇̃ϕ) where T<:Real
    η = T(o.eta)
    return m - η .* ∇̃m, ϕ - η .* ∇̃ϕ
end

"""
    sample(::NaturalDescent, state, i)

Sample from the variational distribution on the Euclidean manifold.
"""
function sample(::NaturalDescent{FullNormal,EuclidianManifold}, state, i::Int)
    m, L = state.q
    return L * state.epsilon[i] + m
end

function sample(::NaturalDescent{DiagNormal,EuclidianManifold}, state, i::Int)
    m, ϕ = state.q
    return softplus.(ϕ) .* state.epsilon[i] + m
end
