"""
Eigenspace Variational Online Newton (EVON) from "SOAP-Bubbles: Structured Weight
Uncertainty for Neural Networks" (Minut et al., 2026).

EVON runs the diagonal variational method IVON (Shen et al., 2024) inside the
eigenbasis of SOAP's Kronecker-factored preconditioner (Vyas et al., 2025). A
diagonal Gaussian `N(w | m, diag(σ²))` is learned over a rotated coordinate
`w`, where `θ = P w` with the orthonormal rotation `P = Q_R ⊗ Q_L`. This maps
the diagonal posterior to a *structured* (block-diagonal, Kronecker-factored)
covariance in the original weight space (Eq. 8):

    q(θ) = N(vec(M), (Q_R ⊗ Q_L) diag(vec(V)) (Q_R ⊗ Q_L)ᵀ).

For a single 2D weight matrix `Θ ∈ ℝ^{d×k}`, EVON tracks the SOAP preconditioner
`L = EMA(G Gᵀ) ∈ ℝ^{d×d}` and `R = EMA(Gᵀ G) ∈ ℝ^{k×k}`, their eigenbases
`Q_L, Q_R`, an eigenspace Hessian EMA `H`, and a projected-gradient momentum `Ḡ`.

This first implementation targets a single 2D matrix parameter per leaf (the
setting in which the SOAP-Bubble rotation is defined); other parameter shapes
are not yet supported.
"""

"""
    EVON(eta, beta=(0.9, 0.999, 0.95); delta=1, zeta=1, init_scale=1, clip=Inf, precond_freq=10)

Construct the EVON optimisation rule.

# Arguments
- `eta`: mean-update learning rate `α` (Alg. 2, line 8).
- `beta = (β₁, β₂, β₃)`: EMA decay rates for, respectively, the projected-gradient
  momentum `Ḡ` (line 5), the eigenspace Hessian `H` (line 6), and the SOAP
  preconditioner matrices `L`, `R` (lines 9–10).

# Keywords
- `delta`: prior precision / (non-decoupled) weight-decay `δ > 0`. The isotropic
  Gaussian prior is `N(0, (ζδ)⁻¹ I)`.
- `zeta`: dataset scaling `ζ` from the variational objective `ζ Eq[ℓ] + KL`
  (typically the number of data points for pure Bayesian inference).
- `init_scale`: initial value of the eigenspace Hessian `H₀`, controlling the
  initial posterior variance `V₀ = 1/(ζ(H₀ + δ))`.
- `init_mean`: if `true`, warm-start the variational mean `M` at the parameter's
  current value (for variational fine-tuning of a checkpoint); otherwise `M = 0`.
- `hess_clip`: fixed element-wise clip bound on the Hessian estimator `Ĥ` (line 4).
- `hess_clip_ratio`: `γ` for *adaptive* relative Hessian clipping (Sec. 3.2),
  `clip(Ĥ, -γ(H+ϵ), γ(H+ϵ))`, relative to the current Hessian EMA `H`. When finite
  this takes precedence over `hess_clip`. The paper defaults `γ = 10`.
- `clip_eps`: the small constant `ϵ` used by the adaptive Hessian clip.
- `update_clip`: element-wise clip bound on the preconditioned update `ΔM` (line 8).
- `spectral`: if `true`, spectral-clip the update via a quintic Newton–Schulz
  orthogonalisation (Muon-style; Alg. 2 line 8 option), instead of element-wise.
- `squared_grad`: if `true`, use the squared-gradient Hessian estimator `Ĥ = G° ⊙ G°`
  (the SOAP/Adam heuristic — positive, biased, lower variance) instead of the default
  reparameterised estimator `Ĥ = G° ⊙ Z / √V` (Alg. 2, line 4). Stored type-stably as a
  singleton so the choice is resolved at compile time.
- `qr_eig`: if `true`, refresh the eigenbasis with a single warm-started orthogonal-iteration
  step (`Q ← orth(L Q)` via a QR decomposition) instead of an exact `eigen` (Vyas et al., 2025).
  Much cheaper per refresh; the slowly-drifting basis is tracked rather than recomputed.
- `bias_correction`: if `true`, Adam/IVON-style debiasing of the gradient momentum `Ḡ` by
  `1/(1 - β₁ᵗ)` in the mean update, improving early-step behaviour. Default `false` to match
  Alg. 2 (which uses plain EMAs); IVON's Alg. 1 does debias, so this aligns EVON with IVON when on.
- `precond_freq`: number of steps `T` between eigenbasis refreshes `Q_L, Q_R ←
  Eig(L), Eig(R)` (line 11).
"""
# Compile-time flag (a `Static`-style singleton stored in the struct) selecting EVON's
# line-4 Hessian estimator, so `apply!` dispatches on it without a runtime branch.
abstract type HessianEstimator end
struct ReparamHessian <: HessianEstimator end   # Ĥ = G° ⊙ Z / √V (default; unbiased)
struct SquaredGradient <: HessianEstimator end  # Ĥ = G° ⊙ G°      (SOAP/Adam heuristic)

struct EVON{T,H<:HessianEstimator} <: AbstractNaturalRule
    eta::T              # α, mean learning rate
    beta::NTuple{3,T}   # (β₁ momentum, β₂ Hessian EMA, β₃ preconditioner EMA)
    delta::T            # δ, prior precision / weight decay
    zeta::T             # ζ, variational dataset scaling
    init_scale::T       # H₀, initial eigenspace Hessian
    hess_clip::T        # element-wise bound on Ĥ, line 4 (Inf = off)
    hess_clip_ratio::T  # γ for adaptive relative Hessian clip, Sec. 3.2 (Inf = off)
    clip_eps::T         # ϵ in the adaptive Hessian clip
    update_clip::T      # element-wise bound on ΔM, line 8 (Inf = off)
    spectral::Bool         # spectral (Newton–Schulz) update clip, line 8
    init_mean::Bool        # warm-start M at the parameter (vs zero)
    qr_eig::Bool           # QR-based approximate eigendecomposition at refresh (line 11)
    bias_correction::Bool  # debias the gradient momentum Ḡ by 1/(1-β₁ᵗ) (off = faithful to Alg. 2)
    hessian::H             # Hessian-estimator flag (ReparamHessian / SquaredGradient)
    precond_freq::Int      # T, eigenbasis refresh interval
end

function EVON(eta, beta=(0.9, 0.999, 0.95); delta=1, zeta=1, init_scale=1, init_mean=false,
    hess_clip=Inf, hess_clip_ratio=Inf, clip_eps=1e-8, update_clip=Inf, spectral=false,
    squared_grad=false, qr_eig=false, bias_correction=false, precond_freq=10)
    (delta <= 0) && throw(ErrorException("`delta` (prior precision) must be positive."))
    (zeta <= 0) && throw(ErrorException("`zeta` (dataset scaling) must be positive."))
    (precond_freq < 1) && throw(ErrorException("`precond_freq` must be a positive integer."))

    T = eltype(eta)
    hessian = squared_grad ? SquaredGradient() : ReparamHessian()
    return EVON{T,typeof(hessian)}(eta, T.(beta), T(delta), T(zeta), T(init_scale), T(hess_clip),
        T(hess_clip_ratio), T(clip_eps), T(update_clip), spectral, init_mean, qr_eig, bias_correction, hessian, precond_freq)
end

# EVON needs a matrix to form the Kronecker-factored rotation, so it does not
# apply to 1D parameters (biases, norms, embeddings). Following Minut et al. (2026,
# App. E.3), those should be optimised with AdamW (Optimisers.jl) or — for a diagonal
# variational treatment — IVON, by assigning a different rule to those leaves; `sample`
# returns the point estimate for any leaf whose rule is not an `AbstractNaturalRule`.
Optimisers.init(::EVON, x::AbstractVector) = throw(ErrorException(
    "EVON does not support 1-dimensional parameters (biases, norms, embeddings). " *
    "Following Minut et al. (2026), optimise these with AdamW or IVON by assigning a " *
    "different rule to those leaves (`sample` handles such mixed optimiser trees)."))

# Matricise the parameter to 2D as (first dimension × product of the rest), per SOAP
# practice, so that >2D tensors (e.g. convolution/attention kernels) are supported.
_matricise(x::AbstractArray) = reshape(x, size(x, 1), :)

# TODO: Need to think about GPUArrays etc.
function Optimisers.init(o::EVON, x::AbstractArray{T}) where T<:Real
    Mx = collect(_matricise(x))
    M = o.init_mean ? Mx : zero(Mx)   # warm-start at the parameter, or start at zero
    d, k = size(M)
    return (
        q=(M, fill(T(o.init_scale), d, k)),                            # mean M, eigenspace Hessian H
        precond=(zeros(T, d, d), zeros(T, k, k), Matrix{T}(I, d, d), Matrix{T}(I, k, k)), # L, R, Q_L, Q_R
        momentum=zero(M),                                              # projected-gradient momentum Ḡ
        epsilon=[zero(M)],                                             # standard-normal noise samples Z
        t=0,                                                          # step counter (for precond_freq)
        shape=size(x)                                                 # original parameter shape (for reshaping samples)
    )
end

# Per-sample Hessian estimator (Alg. 2, line 4), dispatched on the estimator flag.
# The reparameterised (Stein/Price) estimator is unbiased for the projected Hessian;
# the squared-gradient variant is the SOAP/Adam heuristic (positive, biased).
_hess_estimate(::ReparamHessian, G°, Z, sqrtV) = @. G° * Z / sqrtV
_hess_estimate(::SquaredGradient, G°, Z, sqrtV) = G° .* G°

# Hessian EMA update (Alg. 2, line 6). The reparameterised estimator can be negative, so it
# uses IVON's quadratic positivity correction; the non-negative squared-gradient estimator
# uses the plain SOAP/Adam moving average (no correction needed).
_hess_update(::ReparamHessian, H, Ĥ, β₂, δ) =
    @. β₂ * H + (1 - β₂) * Ĥ + (1 - β₂)^2 / 2 * (Ĥ - H)^2 / (H + δ)
_hess_update(::SquaredGradient, H, Ĥ, β₂, δ) = @. β₂ * H + (1 - β₂) * Ĥ

# Element-wise symmetric clip; `bound = Inf` leaves the argument untouched.
_clip(x, bound) = clamp.(x, -bound, bound)

# Eigenbasis (orthonormal eigenvectors) of a symmetric matrix.
_eigbasis(A::AbstractMatrix) = eigen(Symmetric(A)).vectors

# One warm-started orthogonal-iteration step as a cheap approximation to `Eig(A)`
# (Vyas et al., 2025): for symmetric PSD `A`, `orth(A·Q)` rotates `Q` towards `A`'s
# eigenbasis. Starting from the previous basis, a single QR step tracks the slowly
# drifting eigenbasis without an iterative eigensolver.
_qr_eigbasis(A::AbstractMatrix, Q::AbstractMatrix) = Matrix(qr(A * Q).Q)

# Quintic Newton–Schulz orthogonalisation (Jordan et al., 2024b / Muon): returns an
# approximation to the orthogonal polar factor `UVᵀ` of `G` (all singular values ≈ 1),
# i.e. a spectral clip of the update. The iteration acts on the row space, so we
# transpose to keep the smaller dimension first, and normalise by the Frobenius norm.
function _newton_schulz(G::AbstractMatrix{T}; steps::Int=5) where T<:Real
    a, b, c = T(3.4445), T(-4.7750), T(2.0315)
    X = G ./ (norm(G) + eps(T))
    transposed = size(X, 1) > size(X, 2)
    transposed && (X = permutedims(X))
    for _ in 1:steps
        A = X * transpose(X)
        B = b .* A .+ c .* (A * A)
        X = a .* X .+ B * X
    end
    transposed && (X = permutedims(X))
    return X
end

Optimisers.apply!(o::EVON, state, x::AbstractArray{T}, dx::AbstractArray{T}) where T<:Real =
    Optimisers.apply!(o, state, x, [dx])

"""
    Optimisers.apply!(o::EVON, state, x, dx)

Perform one EVON step (Alg. 2) for a matrix parameter (>2D tensors are matricised
to 2D as first-dimension × rest). `dx` is a vector of per-sample loss gradients
`G = ∇ℓ(Θ)` evaluated at weights `Θ` drawn from the current SOAP-Bubble posterior
(see [`sample`](@ref)). As in the rest of this package, the model parameter `x` is
left unchanged; the variational state lives entirely in `state` and weights are
materialised via `sample`.
"""
function Optimisers.apply!(o::EVON, state, ::AbstractArray{T}, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    β₁, β₂, β₃ = o.beta
    δ, ζ, α = o.delta, o.zeta, o.eta
    M, H = state.q
    L, R, QL, QR = state.precond
    Ḡ = state.momentum
    Zs = state.epsilon
    t = state.t + 1

    # Matricise the per-sample gradients to match the (2D) variational state.
    G2 = [_matricise(g) for g in dx]

    # Eigenspace posterior variance V = 1/(ζ(H + δ)) and its square root (line 1).
    V = @. 1 / (ζ * (H + δ))
    sqrtV = sqrt.(V)

    # Project per-sample gradients into the current eigenbasis: G° = Q_Lᵀ G Q_R (line 3).
    G°s = [transpose(QL) * G * QR for G in G2]
    Ḡ° = mean(G°s)

    # Hessian estimator (line 4): reparameterised `G° ⊙ Z / √V` or squared-gradient `G° ⊙ G°`.
    Ĥ = mean(map((G°, Z) -> _hess_estimate(o.hessian, G°, Z, sqrtV), G°s, Zs))
    # Clip the Hessian estimator: adaptive (relative to the current H) if a ratio γ is
    # given (Sec. 3.2), otherwise a fixed element-wise bound (both default to no-op).
    Ĥ = isfinite(o.hess_clip_ratio) ?
        clamp.(Ĥ, .-o.hess_clip_ratio .* (H .+ o.clip_eps), o.hess_clip_ratio .* (H .+ o.clip_eps)) :
        _clip(Ĥ, o.hess_clip)

    # Gradient momentum (line 5) and Hessian EMA (line 6; correction depends on the estimator).
    Ḡ′ = @. β₁ * Ḡ + (1 - β₁) * Ḡ°
    H′ = _hess_update(o.hessian, H, Ĥ, β₂, δ)

    # Preconditioned mean update with the prior/weight-decay term (lines 7–8). Optionally
    # debias the gradient momentum by 1/(1-β₁ᵗ) (Adam/IVON-style; off by default per Alg. 2).
    Ḡu = o.bias_correction ? Ḡ′ ./ (1 - β₁^t) : Ḡ′
    Mproj = transpose(QL) * M * QR
    U = @. (Ḡu + δ * Mproj) / (H′ + δ)
    ΔM = QL * U * transpose(QR)
    # Clip the update: spectral (Newton–Schulz orthogonalisation) or element-wise (line 8).
    ΔM = o.spectral ? _newton_schulz(ΔM) : _clip(ΔM, o.update_clip)
    M′ = M .- α .* ΔM

    # SOAP preconditioner EMAs from the raw (unprojected) gradient (lines 9–10).
    G = mean(G2)
    L′ = β₃ .* L .+ (1 - β₃) .* (G * transpose(G))
    R′ = β₃ .* R .+ (1 - β₃) .* (transpose(G) * G)

    # Every `precond_freq` steps, refresh the eigenbasis and reproject the
    # momentum into the new basis (line 11): Ḡ ← Q_Lⁿᵀ Q_L Ḡ Q_Rᵀ Q_Rⁿ.
    QL′, QR′, Ḡ′ = if iszero(t % o.precond_freq)
        QLn = o.qr_eig ? _qr_eigbasis(L′, QL) : _eigbasis(L′)
        QRn = o.qr_eig ? _qr_eigbasis(R′, QR) : _eigbasis(R′)
        Ḡn = transpose(QLn) * QL * Ḡ′ * transpose(QR) * QRn
        (QLn, QRn, Ḡn)
    else
        (QL, QR, Ḡ′)
    end

    state′ = (
        q=(M′, H′),
        precond=(L′, R′, QL′, QR′),
        momentum=Ḡ′,
        epsilon=Zs,
        t=t,
        shape=state.shape
    )

    return state′, nothing
end

"""
    sample(o::EVON, state, i)

Draw a weight matrix from the SOAP-Bubble posterior using the `i`-th stored
standard-normal noise matrix `Z`:

    Θ = M + Q_L (√V ⊙ Z) Q_Rᵀ,    V = 1/(ζ(H + δ)),

which realises `N(vec(M), (Q_R ⊗ Q_L) diag(vec(V)) (Q_R ⊗ Q_L)ᵀ)` (Eq. 8). The
sample is reshaped back to the original parameter shape (for matricised >2D tensors).
"""
function sample(o::EVON, state, i::Int)
    M, H = state.q
    _, _, QL, QR = state.precond
    V = @. 1 / (o.zeta * (H + o.delta))
    E = sqrt.(V) .* state.epsilon[i]
    Θ = M .+ QL * E * transpose(QR)
    return reshape(Θ, state.shape)
end

"""
    Optimisers.setup(evon::EVON, fallback::Optimisers.AbstractRule, model)

Build a mixed optimiser tree that applies `evon` to every parameter with two or
more dimensions (matrices and higher-order tensors, which EVON matricises) and
`fallback` (e.g. `Optimisers.AdamW()`) to 1D parameters (biases, norms, embeddings).

This is the paper-faithful default of Minut et al. (2026, App. E.3), which uses
AdamW for the 1D tensors and the embedding layer. The resulting tree works with the
standard `Optimisers.update` as well as this package's `sample`/`update_epsilon!`.
"""
function Optimisers.setup(evon::EVON, fallback::Optimisers.AbstractRule, model)
    model isa AbstractVector{<:AbstractArray} && _vector_model_error()
    fmap(model; exclude=Optimisers.isnumeric) do x
        Optimisers.setup(ndims(x) >= 2 ? evon : fallback, x)
    end
end
