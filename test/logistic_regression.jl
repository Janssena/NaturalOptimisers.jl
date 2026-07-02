using NaturalOptimisers
using Test
using Random
using Statistics
using Optimisers
using LinearAlgebra
using LogExpFunctions

# Binary Bayesian logistic regression exactness tests (the paper's Fig. 3 / Cor. 1
# setting). The objective `E_q[NLL] + KL(q ‖ N(0, α⁻¹I))` is convex in the Gaussian
# parameters, so it has a unique optimal Gaussian variational posterior. We compute
# that reference deterministically (Opper–Archambeau fixed point with grid quadrature
# for the 1D expectations) and check that each optimiser recovers it: the full-covariance
# methods recover the optimal full Gaussian, the mean-field methods the optimal diagonal
# Gaussian, and EVON — given the basis that diagonalises the posterior precision — recovers
# the exact full Gaussian, confirming it is a SOAP-Bubble (Cor. 1).

_sigmoid(z) = 1 / (1 + exp(-z))

# E_{z ~ N(μ, s²)}[f(z)] via a fine grid (deterministic, ~exact for smooth bounded f).
function _gauss_expect(f, μ, s2; npts=601)
    s = sqrt(s2)
    zs = range(μ - 9s, μ + 9s; length=npts)
    return sum(f(z) * exp(-(z - μ)^2 / (2s2)) for z in zs) * step(zs) / sqrt(2π * s2)
end

# Optimal Gaussian variational posterior for logistic regression with isotropic prior
# precision `α`. Returns (m*, Σ*, h̄*), where h̄* are the converged expected per-example
# Hessian weights E_q[σ(1-σ)] (so Σ*⁻¹ = Xᵀ diag(h̄*) X + αI).
function optimal_gaussian(X, y, α; meanfield=false, iters=400)
    n, d = size(X)
    m = zeros(d)
    Σ = Matrix(1.0 / α * I, d, d)
    h̄ = zeros(n)
    for _ in 1:iters
        μ = X * m
        s2 = [dot(view(X, i, :), Σ, view(X, i, :)) for i in 1:n]
        σ̄ = [_gauss_expect(_sigmoid, μ[i], s2[i]) for i in 1:n]
        h̄ = [_gauss_expect(z -> _sigmoid(z) * (1 - _sigmoid(z)), μ[i], s2[i]) for i in 1:n]
        Prec = Symmetric(X' * Diagonal(h̄) * X + α * I)
        Σ = meanfield ? Matrix(Diagonal(1 ./ diag(Prec))) : inv(Prec)
        m = m - Σ * (X' * (σ̄ .- y) + α * m)
    end
    return m, Σ, h̄
end

# Run `nsteps` of a rule on this problem, drawing `nsamp` samples per step.
function run_method!(rng, rule, state; gradient, nsteps, nsamp)
    dims = size(state.q[1])
    for _ in 1:nsteps
        empty!(state.epsilon)
        for _ in 1:nsamp
            push!(state.epsilon, randn(rng, Float64, dims))
        end
        zs = [NaturalOptimisers.sample(rule, state, s) for s in 1:nsamp]
        dx = [gradient(z) for z in zs]
        state, _ = Optimisers.apply!(rule, state, state.q[1], dx)
    end
    return state
end

full_cov(::LieGroupManifold, q) = q[2] * q[2]'
full_cov(::EuclidianManifold, q) = q[2] * q[2]'
full_cov(::RiemannianManifold, q) = inv(Matrix(q[2]))
diag_var(::LieGroupManifold, q) = q[2] .^ 2
diag_var(::RiemannianManifold, q) = 1 ./ q[2]
diag_var(::EuclidianManifold, q) = softplus.(q[2]) .^ 2

@testset "Binary logistic regression exactness (Fig. 3 / Cor. 1)" begin
    rng = MersenneTwister(2024)
    d, n = 4, 4
    α = 1.0
    # Orthogonal data points (Cor. 1's condition for EVON exactness): rows of a scaled
    # orthogonal matrix. The resulting optimal full Gaussian is genuinely non-diagonal.
    Q = qr(randn(rng, d, d)).Q |> Matrix
    X = Matrix((Q .* [1.5, 1.0, 0.7, 0.5]')')
    y = Float64.(rand(rng, n) .< _sigmoid.(X * randn(rng, d)))

    m_full, Σ_full, h̄ = optimal_gaussian(X, y, α; meanfield=false)
    m_mf, Σ_mf, _ = optimal_gaussian(X, y, α; meanfield=true)
    @test norm(Σ_full - Diagonal(diag(Σ_full))) > 0.05   # reference covariance is non-diagonal

    ∇nll(θ) = X' * (_sigmoid.(X * θ) .- y)        # IVON / EVON: prior handled via δ
    ∇joint(θ) = ∇nll(θ) .+ α .* θ                 # NaturalDescent: prior folded into ℓ

    manifolds = (LieGroupManifold(), RiemannianManifold(), EuclidianManifold())

    @testset "NaturalDescent FullNormal → optimal full Gaussian" begin
        for man in manifolds
            @testset "$(nameof(typeof(man)))" begin
                rule = NaturalDescent(0.01; tau=1.0, scale=1.0, meanfield=false, manifold=man)
                st = Optimisers.init(rule, zeros(d))
                st = run_method!(MersenneTwister(1), rule, st; gradient=∇joint, nsteps=8000, nsamp=64)
                @test st.q[1] ≈ m_full rtol = 5e-2
                @test full_cov(man, st.q) ≈ Σ_full rtol = 5e-2
            end
        end
    end

    @testset "NaturalDescent DiagNormal → optimal mean-field Gaussian" begin
        for man in manifolds
            @testset "$(nameof(typeof(man)))" begin
                rule = NaturalDescent(0.01; tau=1.0, scale=1.0, meanfield=true, manifold=man)
                st = Optimisers.init(rule, zeros(d))
                st = run_method!(MersenneTwister(1), rule, st; gradient=∇joint, nsteps=8000, nsamp=64)
                @test st.q[1] ≈ m_mf rtol = 5e-2
                @test diag_var(man, st.q) ≈ diag(Σ_mf) rtol = 5e-2
            end
        end
    end

    @testset "IVON → optimal mean-field Gaussian" begin
        rule = IVON(0.05, (0.9, 0.999); delta=α, lambda=1.0, init_scale=1.0)
        st = Optimisers.init(rule, zeros(d))
        st = run_method!(MersenneTwister(1), rule, st; gradient=∇nll, nsteps=8000, nsamp=64)
        σ² = 1 ./ (rule.lambda .* (st.q[2] .+ α))
        @test st.q[1] ≈ m_mf rtol = 5e-2
        @test σ² ≈ diag(Σ_mf) rtol = 5e-2
    end

    @testset "EVON (true basis) recovers the exact full Gaussian → SOAP-Bubble (Cor. 1)" begin
        # The weight is a d×1 matrix (output dimension o = 1, so Q_R is the 1×1 identity).
        # Injecting Q_L = eigenvectors of the posterior precision shows the optimal full
        # Gaussian lies in EVON's variational family.
        QL = eigvecs(Symmetric(X' * Diagonal(h̄) * X + α * I))
        rule = EVON(0.3, (0.9, 0.99, 0.95); delta=α, zeta=1.0, init_scale=1.0, precond_freq=10^9)
        st = Optimisers.init(rule, zeros(d, 1))
        st = (q=st.q, precond=(st.precond[1], st.precond[2], Matrix(QL), ones(1, 1)),
            momentum=st.momentum, epsilon=st.epsilon, t=st.t, shape=st.shape)

        st = run_method!(MersenneTwister(1), rule, st; gradient=Θ -> reshape(∇nll(vec(Θ)), d, 1), nsteps=2000, nsamp=64)

        V = 1 ./ (rule.zeta .* (st.q[2] .+ α))
        P = kron(ones(1, 1), Matrix(QL))
        Σ_evon = P * Diagonal(vec(V)) * P'
        @test vec(st.q[1]) ≈ m_full rtol = 5e-2
        @test Σ_evon ≈ Σ_full rtol = 5e-2
    end
end
