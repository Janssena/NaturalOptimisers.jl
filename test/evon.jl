using NaturalOptimisers
using Test
using Random
using Statistics
using Optimisers
using LinearAlgebra

# Run `nsteps` EVON steps on a single matrix parameter, where `gradient(Θ)`
# returns the loss gradient ∇ℓ(Θ). At each step we draw `nsamp` weights from the
# current SOAP-Bubble posterior, evaluate the gradient at each, and apply the
# rule. Returns the final optimiser state.
function run_evon!(rng, rule, state; gradient, nsteps::Int, nsamp::Int=1)
    d, k = size(state.q[1])
    for _ in 1:nsteps
        empty!(state.epsilon)
        for _ in 1:nsamp
            push!(state.epsilon, randn(rng, Float64, d, k))
        end
        Θs = [NaturalOptimisers.sample(rule, state, s) for s in 1:nsamp]
        dx = [gradient(Θ) for Θ in Θs]
        state, _ = Optimisers.apply!(rule, state, state.q[1], dx)
    end
    return state
end

# Reconstruct the original-space posterior covariance of a SOAP-Bubble from an
# EVON state: Σ = (Q_R ⊗ Q_L) diag(vec(V)) (Q_R ⊗ Q_L)ᵀ, with V = 1/(ζ(H+δ)).
function bubble_covariance(rule, state)
    _, H = state.q
    _, _, QL, QR = state.precond
    V = @. 1 / (rule.zeta * (H + rule.delta))
    P = kron(QR, QL)
    return P * Diagonal(vec(V)) * transpose(P)
end

@testset "EVON (Eigenspace Variational Online Newton)" begin

    @testset "State initialisation and shapes" begin
        rule = EVON(0.1; delta=1.0, zeta=1.0, init_scale=2.0, precond_freq=5)
        x = randn(MersenneTwister(1), 4, 3)
        st = Optimisers.init(rule, x)

        M, H = st.q
        L, R, QL, QR = st.precond
        @test size(M) == (4, 3)
        @test size(H) == (4, 3)
        @test all(==(2.0), H)              # H initialised to init_scale
        @test iszero(M)                    # mean initialised to zero(x)
        @test size(L) == (4, 4) && size(R) == (3, 3)
        @test QL == I && QR == I           # eigenbasis starts at the identity
        @test st.t == 0
        @test st.shape == (4, 3)           # original parameter shape is tracked

        # 1D parameters are not supported (the paper uses AdamW for these).
        @test_throws ErrorException Optimisers.init(rule, randn(4))

        # >2D tensors are matricised to 2D (first dim × product of the rest).
        st3 = Optimisers.init(rule, randn(2, 3, 4))
        @test size(st3.q[1]) == (2, 12)
        @test size(st3.precond[1]) == (2, 2) && size(st3.precond[2]) == (12, 12)
        @test st3.shape == (2, 3, 4)
    end

    @testset "Eigenbasis stays orthonormal after refresh" begin
        rng = MersenneTwister(2)
        rule = EVON(0.1; delta=1.0, zeta=1.0, precond_freq=3)
        x = randn(rng, 5, 4)
        st = Optimisers.init(rule, x)
        st = run_evon!(rng, rule, st; gradient=Θ -> Θ .+ 0.1, nsteps=12, nsamp=2)

        _, _, QL, QR = st.precond
        @test transpose(QL) * QL ≈ I atol = 1e-10
        @test transpose(QR) * QR ≈ I atol = 1e-10
    end

    @testset "Reduces to diagonal IVON when the basis is the identity" begin
        # With Q_L = Q_R = I (never refreshed), a single EVON step must equal the
        # diagonal IVON update applied directly in the original space.
        rng = MersenneTwister(3)
        α, β₁, β₂, δ, ζ = 0.3, 0.9, 0.99, 1.0, 1.0
        rule = EVON(α, (β₁, β₂, 0.95); delta=δ, zeta=ζ, init_scale=1.5, precond_freq=10^9)

        d, k = 3, 2
        x = randn(rng, d, k)
        st = Optimisers.init(rule, x)
        M, H = st.q
        Ḡ = st.momentum

        Z = randn(rng, d, k)
        st.epsilon[1] = Z
        Θ = NaturalOptimisers.sample(rule, st, 1)
        G = randn(rng, d, k)                       # an arbitrary loss gradient at Θ

        st′, _ = Optimisers.apply!(rule, st, x, G)
        M′, H′ = st′.q

        # Independent IVON reference (basis = I, so G° = G and E = √V ⊙ Z).
        V = @. 1 / (ζ * (H + δ))
        Θ_ref = M .+ sqrt.(V) .* Z
        @test Θ ≈ Θ_ref
        Ĥ = G .* Z ./ sqrt.(V)
        Ḡ_ref = @. β₁ * Ḡ + (1 - β₁) * G
        H_ref = @. β₂ * H + (1 - β₂) * Ĥ + (1 - β₂)^2 / 2 * (Ĥ - H)^2 / (H + δ)
        U_ref = @. (Ḡ_ref + δ * M) / (H_ref + δ)
        M_ref = M .- α .* U_ref

        @test H′ ≈ H_ref
        @test M′ ≈ M_ref
    end

    @testset "Exact full-Gaussian recovery with the true basis (Thm. 1)" begin
        # Kronecker-Hessian quadratic loss ℓ(Θ) = ½ tr(A_L Θ A_R Θᵀ) + tr(Bᵀ Θ),
        # so ∇ℓ(Θ) = A_L Θ A_R + B and the vec-Hessian is A = A_R ⊗ A_L. The exact
        # variational posterior among all Gaussians is then
        #   μ* = -(A + δI)⁻¹ vec(B),   Σ* = (1/ζ) (A + δI)⁻¹,
        # which is a SOAP-Bubble when the basis P diagonalises A.
        rng = MersenneTwister(7)
        d, k = 3, 2
        δ, ζ = 1.0, 1.0

        UL = qr(randn(rng, d, d)).Q |> Matrix
        UR = qr(randn(rng, k, k)).Q |> Matrix
        AL = Symmetric(UL * Diagonal([3.0, 1.0, 0.5]) * UL')
        AR = Symmetric(UR * Diagonal([2.0, 0.8]) * UR')
        B = randn(rng, d, k)

        A = kron(AR, AL)
        Σ_star = inv(A + δ * I) / ζ
        μ_star = reshape(-(A + δ * I) \ vec(B), d, k)

        gradient(Θ) = AL * Θ * AR + B

        rule = EVON(0.5, (0.9, 0.99, 0.95); delta=δ, zeta=ζ, init_scale=1.0, precond_freq=10^9)
        st = Optimisers.init(rule, zeros(d, k))
        # Inject the exact (frozen) eigenbasis.
        st = (q=st.q, precond=(st.precond[1], st.precond[2], Matrix(UL), Matrix(UR)),
            momentum=st.momentum, epsilon=st.epsilon, t=st.t, shape=st.shape)

        st = run_evon!(rng, rule, st; gradient, nsteps=1500, nsamp=64)

        M, _ = st.q
        Σ_evon = bubble_covariance(rule, st)

        @test M ≈ μ_star rtol = 5e-2
        @test Σ_evon ≈ Σ_star rtol = 5e-2
    end

    @testset "Eigenbasis discovery recovers the posterior (Thm. 2)" begin
        # With B = 0 the mean gradient vanishes (μ* = 0) and the EMA L = EMA(GGᵀ)
        # is driven purely by the posterior fluctuations A_L E[EEᵀ] A_L, whose
        # eigenbasis is U_L (similarly R ↦ U_R). EVON should therefore discover the
        # rotation from scratch and recover Σ* = (1/ζ)(A + δI)⁻¹.
        rng = MersenneTwister(11)
        d, k = 3, 2
        δ, ζ = 1.0, 1.0

        UL = qr(randn(rng, d, d)).Q |> Matrix
        UR = qr(randn(rng, k, k)).Q |> Matrix
        AL = Symmetric(UL * Diagonal([4.0, 1.5, 0.6]) * UL')
        AR = Symmetric(UR * Diagonal([2.5, 0.9]) * UR')

        A = kron(AR, AL)
        Σ_star = inv(A + δ * I) / ζ

        gradient(Θ) = AL * Θ * AR              # B = 0  ⇒  μ* = 0

        rule = EVON(0.3, (0.9, 0.99, 0.99); delta=δ, zeta=ζ, init_scale=1.0, precond_freq=20)
        st = Optimisers.init(rule, zeros(d, k))
        st = run_evon!(rng, rule, st; gradient, nsteps=4000, nsamp=128)

        M, _ = st.q
        Σ_evon = bubble_covariance(rule, st)

        # The discovered rotation P = Q_R ⊗ Q_L should approximately diagonalise the
        # true Hessian A (this is exactly the simultaneous-diagonalisability condition
        # of Thm. 2 / Lemma 3).
        _, _, QL, QR = st.precond
        P = kron(QR, QL)
        PAP = transpose(P) * A * P
        @test norm(PAP - Diagonal(diag(PAP))) / norm(diag(PAP)) < 0.15

        # Mean stays at zero, and the structured covariance (basis learned from
        # scratch) matches the exact posterior up to MC noise.
        @test norm(M) < 5e-2
        @test sort(eigvals(Σ_evon)) ≈ sort(eigvals(Σ_star)) rtol = 5e-2
        @test Σ_evon ≈ Σ_star rtol = 1e-1
    end

    @testset ">2D tensors are matricised and recovered" begin
        # A 3D parameter is matricised to (size(x,1) × rest); the SOAP-Bubble is then
        # a posterior over the flattened tensor. We define a Kronecker-Hessian quadratic
        # in the matricised space, inject the true (frozen) basis, and check that the
        # mean and structured covariance are recovered and that samples keep the 3D shape.
        rng = MersenneTwister(23)
        dims = (2, 2, 3)          # matricises to d × k = 2 × 6
        d, k = dims[1], prod(dims[2:end])
        δ, ζ = 1.0, 1.0

        UL = qr(randn(rng, d, d)).Q |> Matrix
        UR = qr(randn(rng, k, k)).Q |> Matrix
        AL = Symmetric(UL * Diagonal([2.5, 0.7]) * UL')
        AR = Symmetric(UR * Diagonal([3.0, 2.0, 1.2, 0.8, 0.5, 0.3]) * UR')
        B = randn(rng, d, k)

        A = kron(AR, AL)
        Σ_star = inv(A + δ * I) / ζ
        μ_star = reshape(-(A + δ * I) \ vec(B), d, k)

        # Gradient receives a 3D tensor; matricise, apply the quadratic, reshape back.
        gradient(Θ) = reshape(AL * reshape(Θ, d, k) * AR + B, dims)

        rule = EVON(0.5, (0.9, 0.99, 0.95); delta=δ, zeta=ζ, init_scale=1.0, precond_freq=10^9)
        st = Optimisers.init(rule, zeros(dims))
        @test size(st.q[1]) == (d, k)
        st = (q=st.q, precond=(st.precond[1], st.precond[2], Matrix(UL), Matrix(UR)),
            momentum=st.momentum, epsilon=st.epsilon, t=st.t, shape=st.shape)

        # Samples must come back in the original 3D shape.
        @test size(NaturalOptimisers.sample(rule, st, 1)) == dims

        st = run_evon!(rng, rule, st; gradient, nsteps=1500, nsamp=64)

        @test st.q[1] ≈ μ_star rtol = 5e-2
        @test bubble_covariance(rule, st) ≈ Σ_star rtol = 5e-2
    end

    @testset "Mixed optimiser tree (EVON on 2D, Adam on 1D)" begin
        # The paper uses AdamW for 1D tensors. Optimisers.jl composition supports this:
        # assign EVON to matrix leaves and Adam to vector leaves. `sample` returns a
        # SOAP-Bubble draw for the EVON leaf and the point estimate for the Adam leaf.
        rng = MersenneTwister(31)
        model = (W=randn(rng, 4, 3), b=randn(rng, 4))
        tree = (
            W=Optimisers.setup(EVON(0.1; delta=1.0, zeta=1.0), model.W),
            b=Optimisers.setup(Optimisers.Adam(0.1), model.b)
        )

        s = sample(rng, model, tree)
        @test size(s.W) == (4, 3)
        @test s.b == model.b                     # 1D Adam leaf: point estimate, unchanged
        @test s.W != model.W                     # 2D EVON leaf: a perturbed sample

        # Drawing multiple samples perturbs only the EVON leaf.
        ss = sample(rng, model, tree; num_samples=3)
        @test length(ss) == 3
        @test all(si -> si.b == model.b, ss)
    end

    @testset "Tree-level Optimisers.update (mixed rules, multi-sample)" begin
        # The full Optimisers training API: a vector of per-sample gradient trees is applied,
        # with variational leaves receiving the whole batch and ordinary leaves the mean.
        rng = MersenneTwister(45)
        model = (W=randn(rng, 4, 3), b=randn(rng, 3))
        tree = (
            W=Optimisers.setup(EVON(0.1; delta=1.0, zeta=1.0), model.W),
            b=Optimisers.setup(Optimisers.Adam(0.1), model.b)
        )

        ss = sample(rng, model, tree; num_samples=2)
        grads = [(W=s.W, b=s.b) for s in ss]      # vector of per-sample gradient trees
        tree2, model2 = Optimisers.update(tree, model, grads)

        @test tree2.W.state.t == 1                # EVON advanced exactly one step
        @test model2.W == model.W                 # variational mean lives in state; param untouched
        @test model2.b != model.b                 # Adam updated the 1D leaf
        @test tree2.b.state[3] != tree.b.state[3] # Adam momentum (βᵗ) advanced

        # Single-gradient path goes through the standard Optimisers.update and also works.
        tree3, _ = Optimisers.update(tree, model, grads[1])
        @test tree3.W.state.t == 1
    end

    @testset "Mixed setup: EVON for ≥2D, fallback rule for 1D" begin
        # Optimisers.setup(evon, fallback, model) routes ≥2D params to EVON and 1D
        # params to the fallback rule (paper-faithful: AdamW for biases/norms/embeddings).
        rng = MersenneTwister(51)
        model = (W=randn(rng, 4, 3), b=randn(rng, 4), K=randn(rng, 2, 3, 4), s=randn(rng, 5))
        tree = Optimisers.setup(EVON(0.1; delta=1.0, zeta=1.0), Optimisers.Adam(0.1), model)

        @test tree.W.rule isa EVON
        @test tree.K.rule isa EVON                 # 3D tensor → EVON (matricised)
        @test tree.b.rule isa Optimisers.Adam      # 1D → fallback
        @test tree.s.rule isa Optimisers.Adam

        # sample: SOAP-Bubble draws for the EVON leaves (original shapes), point
        # estimates for the fallback leaves.
        s1 = sample(rng, model, tree)
        @test size(s1.W) == (4, 3) && size(s1.K) == (2, 3, 4)
        @test s1.b == model.b && s1.s == model.s

        # end-to-end update advances both EVON leaves and the Adam leaves.
        g = (W=model.W, b=model.b, K=model.K, s=model.s)
        tree2, model2 = Optimisers.update(tree, model, g)
        @test tree2.W.state.t == 1 && tree2.K.state.t == 1
        @test model2.W == model.W                  # variational mean lives in state
        @test model2.b != model.b                  # Adam updated the 1D leaf
    end

    @testset "Warm-start and option plumbing" begin
        rng = MersenneTwister(41)
        W = randn(rng, 3, 4)
        @test Optimisers.init(EVON(0.1; init_mean=true), W).q[1] == reshape(W, 3, 4)
        @test iszero(Optimisers.init(EVON(0.1; init_mean=false), W).q[1])
    end

    @testset "Element-wise update clipping bounds the step" begin
        rng = MersenneTwister(43)
        α, bound = 1.0, 0.01
        rule = EVON(α; delta=1.0, zeta=1.0, update_clip=bound, precond_freq=10^9)
        st = Optimisers.init(rule, zeros(3, 2))
        st.epsilon[1] = randn(rng, 3, 2)
        st2, _ = Optimisers.apply!(rule, st, zeros(3, 2), 1e3 .* randn(rng, 3, 2))
        @test maximum(abs, st2.q[1]) <= α * bound + 1e-9   # |M'| = α|ΔM| ≤ α·bound
    end

    @testset "Adaptive Hessian clipping shrinks the Hessian estimate" begin
        # With all-positive Ĥ (Z and G aligned), clamping to γ(H+ϵ) reduces every
        # entry of the resulting Hessian EMA relative to the unclipped step.
        rng = MersenneTwister(44)
        Z = abs.(randn(rng, 3, 2)) .+ 0.5
        G = 50 .* (abs.(randn(rng, 3, 2)) .+ 0.5)
        base = EVON(0.1; delta=1.0, zeta=1.0, init_scale=1.0, precond_freq=10^9)
        clipped = EVON(0.1; delta=1.0, zeta=1.0, init_scale=1.0, hess_clip_ratio=2.0, precond_freq=10^9)

        s1 = Optimisers.init(base, zeros(3, 2)); s1.epsilon[1] = copy(Z)
        s2 = Optimisers.init(clipped, zeros(3, 2)); s2.epsilon[1] = copy(Z)
        h1 = Optimisers.apply!(base, s1, zeros(3, 2), copy(G))[1].q[2]
        h2 = Optimisers.apply!(clipped, s2, zeros(3, 2), copy(G))[1].q[2]

        @test all(h2 .<= h1 .+ 1e-9)        # clipping never increases the Hessian estimate
        @test maximum(h2) < maximum(h1)     # and strictly shrinks the large entries
    end

    @testset "Spectral (Newton–Schulz) update clipping" begin
        rng = MersenneTwister(42)
        # NS orthogonalisation pulls singular values towards 1 and preserves shape.
        G = randn(rng, 6, 4)
        X = NaturalOptimisers._newton_schulz(G)
        @test size(X) == size(G)
        @test all(s -> 0.5 < s < 1.5, svdvals(X))
        # An EVON step with spectral clipping runs and stays finite.
        rule = EVON(0.1; delta=1.0, zeta=1.0, spectral=true, precond_freq=10^9)
        st = Optimisers.init(rule, zeros(4, 3))
        st.epsilon[1] = randn(rng, 4, 3)
        st2, _ = Optimisers.apply!(rule, st, zeros(4, 3), randn(rng, 4, 3))
        @test all(isfinite, st2.q[1])
    end

    @testset "Squared-gradient Hessian variant" begin
        rng = MersenneTwister(61)
        # The estimator choice is encoded in the type (a singleton field), so the two
        # rules have distinct concrete types and the estimator dispatches statically.
        r_def = EVON(0.1; delta=1.0, zeta=1.0)
        r_sq = EVON(0.1; delta=1.0, zeta=1.0, squared_grad=true)
        @test r_def.hessian isa NaturalOptimisers.ReparamHessian
        @test r_sq.hessian isa NaturalOptimisers.SquaredGradient
        @test typeof(r_def) != typeof(r_sq)
        G = randn(rng, 3, 2); Z = randn(rng, 3, 2); sV = ones(3, 2)
        @test (@inferred NaturalOptimisers._hess_estimate(r_sq.hessian, G, Z, sV)) == G .* G

        # With basis = I, one step uses Ĥ = G² and a plain (correction-free) Hessian EMA.
        β₂, δ, h0 = 0.99, 1.0, 1.5
        rule = EVON(0.5, (0.9, β₂, 0.95); delta=δ, zeta=1.0, init_scale=h0, squared_grad=true, precond_freq=10^9)
        st = Optimisers.init(rule, zeros(3, 2)); st.epsilon[1] = randn(rng, 3, 2)
        Gs = randn(rng, 3, 2)
        st2, _ = Optimisers.apply!(rule, st, zeros(3, 2), Gs)
        @test st2.q[2] ≈ @. β₂ * h0 + (1 - β₂) * Gs^2     # plain EMA of G², no correction

        # The squared-gradient estimator is biased for the variance, but the mean update is
        # unaffected, so the posterior MEAN still converges to μ* on the Kronecker quadratic.
        d, k = 3, 2
        UL = qr(randn(rng, d, d)).Q |> Matrix
        UR = qr(randn(rng, k, k)).Q |> Matrix
        AL = Symmetric(UL * Diagonal([3.0, 1.0, 0.5]) * UL')
        AR = Symmetric(UR * Diagonal([2.0, 0.8]) * UR')
        B = randn(rng, d, k)
        A = kron(AR, AL)
        μ_star = reshape(-(A + I) \ vec(B), d, k)
        gradient(Θ) = AL * Θ * AR + B

        rule2 = EVON(0.3, (0.9, 0.99, 0.95); delta=1.0, zeta=1.0, init_scale=1.0, squared_grad=true, precond_freq=10^9)
        st = Optimisers.init(rule2, zeros(d, k))
        st = (q=st.q, precond=(st.precond[1], st.precond[2], Matrix(UL), Matrix(UR)),
            momentum=st.momentum, epsilon=st.epsilon, t=st.t, shape=st.shape)
        st = run_evon!(rng, rule2, st; gradient, nsteps=1500, nsamp=64)
        @test st.q[1] ≈ μ_star rtol = 5e-2
    end

    @testset "Optional gradient-momentum bias correction" begin
        # With basis = I, one step's mean update uses Ḡ′ (default) or the debiased
        # Ḡ′/(1-β₁ᵗ) (at t = 1, that is the full gradient G rather than (1-β₁)G).
        rng = MersenneTwister(81)
        α, β₁, β₂, δ, h0 = 0.3, 0.9, 0.99, 1.0, 1.5
        d, k = 3, 2
        Z = randn(rng, d, k); G = randn(rng, d, k)
        for bc in (false, true)
            rule = EVON(α, (β₁, β₂, 0.95); delta=δ, zeta=1.0, init_scale=h0,
                bias_correction=bc, precond_freq=10^9)
            st = Optimisers.init(rule, zeros(d, k)); st.epsilon[1] = copy(Z)
            st2, _ = Optimisers.apply!(rule, st, zeros(d, k), copy(G))

            H = fill(h0, d, k); V = @. 1 / (1.0 * (H + δ)); Ĥ = G .* Z ./ sqrt.(V)
            Ḡ′ = @. (1 - β₁) * G                                   # momentum from Ḡ₀ = 0
            H′ = @. β₂ * H + (1 - β₂) * Ĥ + (1 - β₂)^2 / 2 * (Ĥ - H)^2 / (H + δ)
            Ḡu = bc ? Ḡ′ ./ (1 - β₁) : Ḡ′                          # t = 1
            M_ref = @. 0 - α * (Ḡu / (H′ + δ))
            @test st2.q[1] ≈ M_ref
        end
    end

    @testset "QR-based approximate eigendecomposition" begin
        rng = MersenneTwister(71)
        # One warm-started QR step is orthogonal iteration: iterating it diagonalises a
        # fixed symmetric PSD matrix (with well-separated eigenvalues for fast, deterministic
        # convergence) and keeps the basis orthonormal.
        U = qr(randn(rng, 5, 5)).Q |> Matrix
        A = Symmetric(U * Diagonal([16.0, 8.0, 4.0, 2.0, 1.0]) * U')
        Q = Matrix{Float64}(I, 5, 5)
        for _ in 1:30
            Q = NaturalOptimisers._qr_eigbasis(A, Q)
        end
        @test Q' * Q ≈ I atol = 1e-10
        PAP = Q' * A * Q
        @test norm(PAP - Diagonal(diag(PAP))) / norm(diag(PAP)) < 1e-4

        # With the QR approximation EVON still discovers the basis and recovers Σ*
        # (Thm. 2 problem, B = 0), at accuracy comparable to the exact eigensolver.
        d, k = 3, 2
        δ, ζ = 1.0, 1.0
        UL = qr(randn(rng, d, d)).Q |> Matrix
        UR = qr(randn(rng, k, k)).Q |> Matrix
        AL = Symmetric(UL * Diagonal([4.0, 1.5, 0.6]) * UL')
        AR = Symmetric(UR * Diagonal([2.5, 0.9]) * UR')
        Σ_star = inv(kron(AR, AL) + δ * I) / ζ
        gradient(Θ) = AL * Θ * AR

        rule = EVON(0.3, (0.9, 0.99, 0.99); delta=δ, zeta=ζ, init_scale=1.0, precond_freq=20, qr_eig=true)
        st = Optimisers.init(rule, zeros(d, k))
        st = run_evon!(MersenneTwister(11), rule, st; gradient, nsteps=4000, nsamp=128)
        Σ_evon = bubble_covariance(rule, st)

        @test sort(eigvals(Σ_evon)) ≈ sort(eigvals(Σ_star)) rtol = 5e-2
        @test Σ_evon ≈ Σ_star rtol = 1e-1
    end
end
