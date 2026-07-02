using NaturalOptimisers
using Test
using Random
using Statistics
using Optimisers
using LinearAlgebra
using LogExpFunctions

# Run `nsteps` NaturalDescent steps on a parameter vector, where `gradient(z)`
# returns the gradient of the potential ℓ(z) = -log p(y, z). Each step draws
# `nsamp` reparameterised samples, evaluates the gradient at each, and applies
# the rule. Returns the final optimiser state.
function run_nd!(rng, rule, state; gradient, nsteps::Int, nsamp::Int=1)
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

# Reconstruct the covariance / variance from a converged state, per manifold's
# scale parameterisation: LieGroup/Euclidian store a Cholesky factor L (Σ = LLᵀ),
# Riemannian stores the precision S (Σ = S⁻¹). DiagNormal stores σ (LieGroup),
# the precision s (Riemannian), or ϕ with σ = softplus(ϕ) (Euclidian).
full_cov(::LieGroupManifold, q) = q[2] * q[2]'
full_cov(::EuclidianManifold, q) = q[2] * q[2]'
full_cov(::RiemannianManifold, q) = inv(Matrix(q[2]))
diag_var(::LieGroupManifold, q) = q[2] .^ 2
diag_var(::RiemannianManifold, q) = 1 ./ q[2]
diag_var(::EuclidianManifold, q) = softplus.(q[2]) .^ 2

@testset "NaturalDescent true-posterior recovery (τ = 1)" begin
    # With τ = 1 and a quadratic potential ℓ(z) = ½ zᵀAz + bᵀz (the negative
    # log-joint of a Gaussian model), the exact posterior among all Gaussians is
    #   N(-A⁻¹b, A⁻¹).
    # This is the same problem family used to validate IVON and EVON; here we run
    # the full sample → gradient → apply! loop to convergence for every manifold
    # and both the full-covariance and mean-field families.
    manifolds = (LieGroupManifold(), RiemannianManifold(), EuclidianManifold())
    η, steps, nsamp = 0.01, 8000, 64

    @testset "FullNormal recovers N(-A⁻¹b, A⁻¹)" begin
        # Non-diagonal SPD Hessian: the exact posterior covariance is full.
        rng = MersenneTwister(101)
        U = qr(randn(rng, 3, 3)).Q |> Matrix
        A = Symmetric(U * Diagonal([3.0, 1.5, 0.6]) * U')
        b = randn(rng, 3)
        gradient(z) = A * z + b
        m_star = -(A \ b)
        Σ_star = inv(A)

        for man in manifolds
            @testset "$(nameof(typeof(man)))" begin
                rule = NaturalDescent(η; tau=1.0, scale=1.0, meanfield=false, manifold=man)
                st = Optimisers.init(rule, zeros(3))
                st = run_nd!(MersenneTwister(7), rule, st; gradient, nsteps=steps, nsamp=nsamp)

                @test st.q[1] ≈ m_star rtol = 3e-2
                @test full_cov(man, st.q) ≈ Σ_star rtol = 5e-2
            end
        end
    end

    @testset "DiagNormal recovers the mean-field posterior" begin
        # Diagonal Hessian ⇒ the exact posterior is itself diagonal, so the
        # mean-field families recover it exactly: m* = -b ./ a, σ²* = 1 ./ a.
        rng = MersenneTwister(202)
        a = [3.0, 1.0, 0.5, 2.0]
        b = randn(rng, 4)
        gradient(z) = a .* z + b
        m_star = -b ./ a
        var_star = 1 ./ a

        for man in manifolds
            @testset "$(nameof(typeof(man)))" begin
                rule = NaturalDescent(η; tau=1.0, scale=1.0, meanfield=true, manifold=man)
                st = Optimisers.init(rule, zeros(4))
                st = run_nd!(MersenneTwister(7), rule, st; gradient, nsteps=steps, nsamp=nsamp)

                @test st.q[1] ≈ m_star rtol = 3e-2
                @test diag_var(man, st.q) ≈ var_star rtol = 5e-2
            end
        end
    end
end
