"""
Calculation of the natural gradient using the Lie-group method from "The Lie-Group Bayesian Learning Rule" (Kiral et al., 2023).
"""

"""
    natgrad(o::NaturalDescent{FullNormal,LieGroupManifold}, state, dx)

Calculate the Lie-Group Natural Gradient for a Full-Rank Gaussian parameterized by the mean `m` 
and Cholesky scale matrix `L` (where Σ = L * Lᵀ).

### Mathematical Justification (Kiral et al., 2023):
Under the Affine Lie group action, the variational distribution is parameterized by the transformation:
    θ = A * ϵ + b
where b ≡ m is the mean (translation parameter) and A ≡ L is the scale matrix.

The tangent vectors in the Lie algebra correspond to (Eq 13, Kiral et al., 2023):
* Mean Tangent Vector (V):
    V = Aᵀ * E_q[∇_θ ℓ] = Lᵀ * E_q[∇_θ ℓ]
* Variance Tangent Vector (U):
    U = Aᵀ * E_q[∇_θ ℓ * ϵᵀ] - τ * I = Lᵀ * E_q[∇_θ ℓ * ϵᵀ] - τ * I
"""
function natgrad(o::NaturalDescent{FullNormal,LieGroupManifold}, state, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    τ = T(o.tau)
    _, L = state.q
    Lᵀ = transpose(L)
    V = Lᵀ * mean(dx) # ∇̃m
    U = Lᵀ * mean(dx .* transpose.(state.epsilon)) - τ * I # ∇̃L

    return V, U
end

"""
    natgrad(o::NaturalDescent{DiagNormal,LieGroupManifold}, state, dx)

Calculate the Lie-Group Natural Gradient for a Diagonal Gaussian parameterized by the mean `m` 
and standard deviation vector `σ`.

### Mathematical Justification:
For the diagonal (meanfield) case, the scale matrix A = diag(σ) is diagonal.
The Lie algebra elements simplify to coordinate-wise (Hadamard) products:
* Mean Tangent Vector (V):
    V = σ ⊙ E_q[∇_θ ℓ]
* Variance Tangent Vector (U):
    U = σ ⊙ E_q[∇_θ ℓ ⊙ ϵ] - τ
"""
function natgrad(o::NaturalDescent{DiagNormal,LieGroupManifold}, state, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    τ = T(o.tau)
    _, σ = state.q
    V = σ .* mean(dx) # ∇̃m
    U = σ .* mean(map(.*, dx, state.epsilon)) .- τ # ∇̃σ

    return V, U
end

"""
    update(o::NaturalDescent{FullNormal,LieGroupManifold}, m, L, V, U)

Perform the parameter update step on the Full-Rank Gaussian Lie-Group manifold.

### Mathematical Justification:
Using the exponential map of the GL_d(ℝ) group for the scale parameter:
    m_new = m - η * L * V (linearized retraction)
    L_new = L * exp(-η * U) (exact exponential map)
"""
function update(o::NaturalDescent{FullNormal,LieGroupManifold}, m::AbstractArray{T}, L::LowerTriangular, V, U) where T<:Real
    η = T(o.eta)
    return m - η * L * V, LowerTriangular(L * exp(-η * U))
end

"""
    update(o::NaturalDescent{DiagNormal,LieGroupManifold}, m, σ, V, U)

Perform the parameter update step on the Diagonal Gaussian Lie-Group manifold using the exact coupled 
exponential map for the Affine group.

### Mathematical Justification:
The exact coupled exponential map for the diagonal Affine group is (Eq 16/31, Kiral et al., 2023):
    m_new = m + σ ⊙ [(exp(-η * U) - I) / U] ⊙ (-η * V)
    σ_new = σ ⊙ exp(-η * U)

This mathematically exact coupling prevents overshooting when the scale σ is changing rapidly.
"""
function update(o::NaturalDescent{DiagNormal,LieGroupManifold}, m, σ::AbstractArray{T}, V, U) where {T<:Real}
    η = T(o.eta)
    exp_neg_ηU = exp.(-η .* U)
    m_new = m + σ .* (exp_neg_ηU .- 1) ./ U .* -η .* V
    σ_new = σ .* exp_neg_ηU
    return m_new, σ_new
end

"""
    sample(::NaturalDescent, state, i)

Sample from the variational distribution on the Lie-Group manifold.
"""
function sample(::NaturalDescent{FullNormal,LieGroupManifold}, state, i::Int)
    m, L = state.q
    return L * state.epsilon[i] + m
end

function sample(::NaturalDescent{DiagNormal,LieGroupManifold}, state, i::Int)
    m, σ = state.q
    return σ .* state.epsilon[i] + m
end