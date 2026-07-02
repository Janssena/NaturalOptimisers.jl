using NaturalOptimisers
using Test
using Random
using Statistics
using Optimisers
using LinearAlgebra

# Run `nsteps` IVON steps on a parameter vector, where `gradient(θ)` returns the
# loss gradient ∇ℓ̄(θ). Each step draws `nsamp` weights from the current posterior,
# evaluates the gradient at each, and applies the rule. Returns the final state.
function run_ivon!(rng, rule, state; gradient, nsteps::Int, nsamp::Int=1)
    dims = size(state.q[1])
    for _ in 1:nsteps
        empty!(state.epsilon)
        for _ in 1:nsamp
            push!(state.epsilon, randn(rng, Float64, dims))
        end
        θs = [NaturalOptimisers.sample(rule, state, s) for s in 1:nsamp]
        dx = [gradient(θ) for θ in θs]
        state, _ = Optimisers.apply!(rule, state, state.q[1], dx)
    end
    return state
end

# Posterior variance σ² = 1/(λ(h + δ)) implied by an IVON state.
posterior_var(rule, state) = @. 1 / (rule.lambda * (state.q[2] + rule.delta))

# Train a 2-leaf (W, b) model through the shared tree API (setup/sample/update) on
# separable diagonal-quadratic potentials ∇ = a ⊙ θ + b per leaf. Returns the final tree.
function train_ivon_tree(rng, model, tree, aW, bW, ab, bb; nsteps, nsamp)
    for _ in 1:nsteps
        samples = sample(rng, model, tree; num_samples=nsamp)         # vector of nsamp models
        grads = map(s -> (W=aW .* s.W .+ bW, b=ab .* s.b .+ bb), samples)
        tree, _ = Optimisers.update(tree, model, grads)               # pass the vector of grad-trees
    end
    return tree
end

@testset "IVON (Improved Variational Online Newton)" begin

    @testset "State initialisation and shapes" begin
        rule = IVON(0.1; delta=1.0, lambda=2.0, init_scale=0.5)
        x = randn(MersenneTwister(1), 6)
        st = Optimisers.init(rule, x)

        m, h = st.q
        g, β₁ᵗ = st.momentum
        @test size(m) == (6,) && size(h) == (6,)
        @test iszero(m)                    # mean initialised to zero(x)
        @test all(==(0.5), h)              # h initialised to init_scale
        @test iszero(g)
        @test β₁ᵗ == rule.beta[1]          # running β₁ᵗ starts at β₁
        @test length(st.epsilon) == 1

        # IVON is mean-field / element-wise, so any array shape is supported.
        @test Optimisers.init(rule, randn(3, 4)).q[1] |> size == (3, 4)
    end

    @testset "Single-step update matches Alg. 1" begin
        rng = MersenneTwister(3)
        α, β₁, β₂, δ, λ = 0.3, 0.9, 0.99, 1.0, 1.0
        rule = IVON(α, (β₁, β₂); delta=δ, lambda=λ, init_scale=1.5)

        P = 4
        x = randn(rng, P)
        st = Optimisers.init(rule, x)
        m, h = st.q
        g, _ = st.momentum

        ϵ = randn(rng, P)
        st.epsilon[1] = ϵ
        θ = NaturalOptimisers.sample(rule, st, 1)
        ĝ = randn(rng, P)                         # an arbitrary loss gradient at θ

        st′, _ = Optimisers.apply!(rule, st, x, ĝ)
        m′, h′ = st′.q

        # Independent reference implementation of Alg. 1 (lines 2–8).
        σ = @. 1 / sqrt(λ * (h + δ))
        @test θ ≈ m .+ σ .* ϵ
        ĥ = ĝ .* ϵ ./ σ
        g_ref = @. β₁ * g + (1 - β₁) * ĝ
        h_ref = @. β₂ * h + (1 - β₂) * ĥ + (1 - β₂)^2 / 2 * (h - ĥ)^2 / (h + δ)
        ḡ = g_ref ./ (1 - β₁)                     # first step: β₁ᵗ = β₁
        m_ref = m .- α .* (ḡ .+ δ .* m) ./ (h_ref .+ δ)

        @test h′ ≈ h_ref
        @test m′ ≈ m_ref
        @test st′.momentum[2] ≈ β₁^2             # running β₁ᵗ advanced
    end

    @testset "Exact posterior recovery on a diagonal quadratic" begin
        # ℓ̄(θ) = ½ θᵀ diag(a) θ + bᵀθ  ⇒  ∇ℓ̄ = a ⊙ θ + b. With an isotropic prior
        # precision δ, the exact Gaussian posterior is diagonal:
        #   m* = -b ./ (a + δ),   σ²* = 1/(λ(a + δ)),
        # which is exactly representable by IVON's mean-field family.
        rng = MersenneTwister(7)
        δ, λ = 1.0, 1.0
        a = [3.0, 1.0, 0.5, 2.0]
        b = randn(rng, 4)
        gradient(θ) = a .* θ .+ b

        m_star = -b ./ (a .+ δ)
        var_star = 1 ./ (λ .* (a .+ δ))

        rule = IVON(0.2, (0.9, 0.99); delta=δ, lambda=λ, init_scale=1.0)
        st = Optimisers.init(rule, zeros(4))
        st = run_ivon!(rng, rule, st; gradient, nsteps=3000, nsamp=64)

        m, h = st.q
        @test m ≈ m_star rtol = 3e-2
        @test h ≈ a rtol = 5e-2                   # Hessian converges to diag(A)
        @test posterior_var(rule, st) ≈ var_star rtol = 5e-2
    end

    @testset "Exact mean and mean-field variance on a non-diagonal quadratic" begin
        # For a non-diagonal SPD Hessian A, IVON recovers the exact posterior mean
        # m* = -(A + δI)⁻¹ b (the mean update uses the full gradient), while the
        # diagonal variance matches the mean-field optimum σ²* = 1/(λ(diag(A) + δ)).
        rng = MersenneTwister(13)
        δ, λ = 1.0, 1.0
        U = qr(randn(rng, 4, 4)).Q |> Matrix
        A = Symmetric(U * Diagonal([4.0, 2.0, 1.0, 0.5]) * U')
        b = randn(rng, 4)
        gradient(θ) = A * θ .+ b

        m_star = -(A + δ * I) \ b
        var_star = 1 ./ (λ .* (diag(A) .+ δ))

        rule = IVON(0.2, (0.9, 0.99); delta=δ, lambda=λ, init_scale=1.0)
        st = Optimisers.init(rule, zeros(4))
        st = run_ivon!(rng, rule, st; gradient, nsteps=3000, nsamp=64)

        m, h = st.q
        @test m ≈ m_star rtol = 5e-2
        @test h ≈ diag(A) rtol = 5e-2
        @test posterior_var(rule, st) ≈ var_star rtol = 5e-2
    end

    @testset "Warm-start mean (init_mean)" begin
        rng = MersenneTwister(21)
        x = randn(rng, 5)
        @test Optimisers.init(IVON(0.1; init_mean=true), x).q[1] == x
        @test iszero(Optimisers.init(IVON(0.1; init_mean=false), x).q[1])
    end

    @testset "Tree-level training via the shared API" begin
        # IVON works on parameters of any shape, so a whole model can use a single IVON
        # rule through Optimisers.setup / sample / update. Here a two-leaf model with
        # independent diagonal-quadratic potentials must recover each leaf's posterior.
        rng = MersenneTwister(22)
        δ, λ = 1.0, 1.0
        aW = [3.0, 1.0, 0.5]; bW = randn(rng, 3)
        ab = [2.0, 0.8]; bb = randn(rng, 2)
        model = (W=zeros(3), b=zeros(2))
        tree = Optimisers.setup(IVON(0.2, (0.9, 0.99); delta=δ, lambda=λ, init_scale=1.0), model)

        tree = train_ivon_tree(MersenneTwister(7), model, tree, aW, bW, ab, bb; nsteps=3000, nsamp=32)

        # Exact mean-field posterior per leaf: m* = -b/(a+δ), σ²* = 1/(λ(a+δ)).
        @test tree.W.state.q[1] ≈ -bW ./ (aW .+ δ) rtol = 1e-1
        @test tree.b.state.q[1] ≈ -bb ./ (ab .+ δ) rtol = 1e-1
        @test 1 ./ (λ .* (tree.W.state.q[2] .+ δ)) ≈ 1 ./ (aW .+ δ) rtol = 5e-2
        @test 1 ./ (λ .* (tree.b.state.q[2] .+ δ)) ≈ 1 ./ (ab .+ δ) rtol = 5e-2
    end
end
