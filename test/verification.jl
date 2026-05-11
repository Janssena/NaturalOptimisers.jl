using NaturalOptimisers
using Test
using Random
using Statistics
using LogExpFunctions
using Optimisers
using LinearAlgebra

@testset "Mathematical Verification of Natural Gradients (1D)" begin
    rng = Random.MersenneTwister(42)
    N = 1_000_000  # High number of samples to eliminate MC noise
    m_val = randn(rng)    # Non-zero mean initialization
    τ = 1.0
    η = 0.1

    @testset "Quadratic Objective (l(z) = 0.5 * z^2)" begin
        @testset "LieGroupManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=LieGroupManifold())
            state_init = Optimisers.init(rule, [m_val])

            # Manually inject non-zero mean into state
            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = z  # dx = z

            ∇̃m, ∇̃σ = natgrad(rule, state, dx)

            m_actual, σ_vec = state.q
            σ = σ_vec[1]
            V_analytical = σ * m_actual[1]
            U_analytical = σ^2 - τ

            @test ∇̃m[1] ≈ V_analytical rtol = 5e-3
            @test ∇̃σ[1] ≈ U_analytical rtol = 5e-3
        end

        @testset "EuclidianManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=EuclidianManifold())
            state_init = Optimisers.init(rule, [m_val])

            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = z

            ∇̃m, ∇̃ϕ = natgrad(rule, state, dx)

            m_actual, ϕ_vec = state.q
            ϕ = ϕ_vec[1]
            σ = softplus(ϕ)

            ∇̃m_analytical = σ^2 * m_actual[1]
            ∇̃ϕ_analytical = σ * (σ^2 - τ) / (2 * logistic(ϕ))

            @test ∇̃m[1] ≈ ∇̃m_analytical rtol = 5e-3
            @test ∇̃ϕ[1] ≈ ∇̃ϕ_analytical rtol = 5e-3
        end

        @testset "RiemannianManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=RiemannianManifold())
            state_init = Optimisers.init(rule, [m_val])

            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = z

            ∇̃m, ∇̃s = natgrad(rule, state, dx)

            m_actual, s_vec = state.q
            s = s_vec[1]
            σ = 1.0 / sqrt(s)

            ∇̃m_analytical = m_actual[1] / s
            ∇̃s_analytical = (τ * s - 1) - η / 2 * (τ * s - 1)^2 / s

            @test ∇̃m[1] ≈ ∇̃m_analytical rtol = 5e-3
            @test ∇̃s[1] ≈ ∇̃s_analytical rtol = 5e-3
        end
    end

    @testset "Linear Objective (l(z) = c * z)" begin
        c = 0.5
        @testset "LieGroupManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=LieGroupManifold())
            state_init = Optimisers.init(rule, [m_val])

            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            dx = [[c] for _ in 1:N]

            ∇̃m, ∇̃σ = natgrad(rule, state, dx)

            _, σ_vec = state.q
            σ = σ_vec[1]
            V_analytical = σ * c
            U_analytical = -τ

            @test ∇̃m[1] ≈ V_analytical rtol = 5e-3
            @test ∇̃σ[1] ≈ U_analytical rtol = 5e-3
        end

        @testset "EuclidianManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=EuclidianManifold())
            state_init = Optimisers.init(rule, [m_val])

            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            dx = [[c] for _ in 1:N]

            ∇̃m, ∇̃ϕ = natgrad(rule, state, dx)

            _, ϕ_vec = state.q
            ϕ = ϕ_vec[1]
            σ = softplus(ϕ)

            ∇̃m_analytical = σ^2 * c
            ∇̃ϕ_analytical = -τ * σ / (2 * logistic(ϕ))

            @test ∇̃m[1] ≈ ∇̃m_analytical rtol = 5e-3
            @test ∇̃ϕ[1] ≈ ∇̃ϕ_analytical rtol = 5e-3
        end

        @testset "RiemannianManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=RiemannianManifold())
            state_init = Optimisers.init(rule, [m_val])

            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            dx = [[c] for _ in 1:N]

            ∇̃m, ∇̃s = natgrad(rule, state, dx)

            _, s_vec = state.q
            s = s_vec[1]

            ∇̃m_analytical = c / s
            ∇̃s_analytical = (τ * s) - η / 2 * (τ * s)^2 / s

            @test ∇̃m[1] ≈ ∇̃m_analytical rtol = 5e-3
            @test ∇̃s[1] ≈ ∇̃s_analytical rtol = 5e-3
        end
    end

    @testset "Exponential Objective (l(z) = exp(z))" begin
        @testset "LieGroupManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=LieGroupManifold())
            state_init = Optimisers.init(rule, [m_val])

            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [exp.(z_i) for z_i in z]

            ∇̃m, ∇̃σ = natgrad(rule, state, dx)

            m_actual, σ_vec = state.q
            σ = σ_vec[1]
            E = exp(m_actual[1] + 0.5 * σ^2)

            V_analytical = σ * E
            U_analytical = σ^2 * E - τ

            @test ∇̃m[1] ≈ V_analytical rtol = 1e-2
            @test ∇̃σ[1] ≈ U_analytical rtol = 1e-2
        end

        @testset "EuclidianManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=EuclidianManifold())
            state_init = Optimisers.init(rule, [m_val])

            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [exp.(z_i) for z_i in z]

            ∇̃m, ∇̃ϕ = natgrad(rule, state, dx)

            m_actual, ϕ_vec = state.q
            ϕ = ϕ_vec[1]
            σ = softplus(ϕ)
            E = exp(m_actual[1] + 0.5 * σ^2)

            ∇̃m_analytical = σ^2 * E
            ∇̃ϕ_analytical = σ * (σ^2 * E - τ) / (2 * logistic(ϕ))

            @test ∇̃m[1] ≈ ∇̃m_analytical rtol = 1e-2
            @test ∇̃ϕ[1] ≈ ∇̃ϕ_analytical rtol = 1e-2
        end

        @testset "RiemannianManifold" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=RiemannianManifold())
            state_init = Optimisers.init(rule, [m_val])

            state = (
                q=([m_val], state_init.q[2]),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 1) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [exp.(z_i) for z_i in z]

            ∇̃m, ∇̃s = natgrad(rule, state, dx)

            m_actual, s_vec = state.q
            s = s_vec[1]
            σ = 1.0 / sqrt(s)
            E = exp(m_actual[1] + 0.5 * σ^2)

            ∇̃m_analytical = E / s
            Ĝ = τ * s - E
            ∇̃s_analytical = Ĝ - η / 2 * Ĝ^2 / s

            @test ∇̃m[1] ≈ ∇̃m_analytical rtol = 1e-2
            @test ∇̃s[1] ≈ ∇̃s_analytical rtol = 1e-2
        end
    end

    @testset "Multivariate FullNormal Verification (2D Quadratic Objective)" begin
        A = [2.0 0.5; 0.5 1.0]
        b = [-0.5, 1.0]
        m_2d = [0.5, -0.5]
        L_val = LowerTriangular([1.5 0.0; 0.5 0.8])
        S_val = Symmetric(inv(L_val * L_val'))

        @testset "LieGroupManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=LieGroupManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, L_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [A * z_i + b for z_i in z]

            ∇̃m, ∇̃L = natgrad(rule, state, dx)

            V_analytical = L_val' * (A * m_2d + b)
            U_analytical = L_val' * A * L_val - τ * I

            @test ∇̃m ≈ V_analytical rtol = 2e-2
            @test ∇̃L ≈ U_analytical rtol = 2e-2
        end

        @testset "EuclidianManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=EuclidianManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, L_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [A * z_i + b for z_i in z]

            ∇̃m, ∇̃L = natgrad(rule, state, dx)

            ∇̃m_analytical = (L_val * L_val') * (A * m_2d + b)
            Lᵀ∇L = L_val' * A * L_val - τ * I
            ∇̃L_analytical = L_val * (LowerTriangular(Lᵀ∇L) - 0.5 * Diagonal(Lᵀ∇L))

            @test ∇̃m ≈ ∇̃m_analytical rtol = 2e-2
            @test ∇̃L ≈ ∇̃L_analytical rtol = 2e-2
        end

        @testset "RiemannianManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=RiemannianManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, S_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            # S = S_val => Covariance is S_val^-1 = inv(S_val)
            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [A * z_i + b for z_i in z]

            ∇̃m, ∇̃S = natgrad(rule, state, dx)

            ∇̃m_analytical = inv(S_val) * (A * m_2d + b)
            Ĝ = τ * S_val - A
            ∇̃S_analytical = Ĝ - η / 2 * (Ĝ * inv(S_val) * Ĝ)

            @test ∇̃m ≈ ∇̃m_analytical rtol = 2e-2
            @test ∇̃S ≈ ∇̃S_analytical rtol = 2e-2
        end
    end

    @testset "Multivariate FullNormal Verification (2D Linear Objective)" begin
        c = [0.5, -0.5]
        m_2d = [0.5, -0.5]
        L_val = LowerTriangular([1.5 0.0; 0.5 0.8])
        S_val = Symmetric(inv(L_val * L_val'))

        @testset "LieGroupManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=LieGroupManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, L_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            dx = [c for _ in 1:N]

            ∇̃m, ∇̃L = natgrad(rule, state, dx)

            V_analytical = L_val' * c
            U_analytical = -τ * I

            @test ∇̃m ≈ V_analytical rtol = 2e-2
            @test ∇̃L ≈ U_analytical rtol = 2e-2
        end

        @testset "EuclidianManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=EuclidianManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, L_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            dx = [c for _ in 1:N]

            ∇̃m, ∇̃L = natgrad(rule, state, dx)

            ∇̃m_analytical = (L_val * L_val') * c
            ∇̃L_analytical = -0.5 * τ * L_val

            @test ∇̃m ≈ ∇̃m_analytical rtol = 2e-2
            @test ∇̃L ≈ ∇̃L_analytical rtol = 2e-2
        end

        @testset "RiemannianManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=RiemannianManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, S_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            dx = [c for _ in 1:N]

            ∇̃m, ∇̃S = natgrad(rule, state, dx)

            ∇̃m_analytical = inv(S_val) * c
            Ĝ = τ * S_val
            ∇̃S_analytical = Ĝ - η / 2 * (Ĝ * inv(S_val) * Ĝ)

            @test ∇̃m ≈ ∇̃m_analytical rtol = 2e-2
            @test ∇̃S ≈ ∇̃S_analytical rtol = 2e-2
        end
    end

    @testset "Multivariate FullNormal Verification (2D Exponential Objective)" begin
        d = [0.5, 0.2]
        m_2d = [0.5, -0.5]
        L_val = LowerTriangular([1.5 0.0; 0.5 0.8])
        S_val = Symmetric(inv(L_val * L_val'))

        @testset "LieGroupManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=LieGroupManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, L_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [exp(d' * z_i) * d for z_i in z]

            ∇̃m, ∇̃L = natgrad(rule, state, dx)

            E = exp(d' * m_2d + 0.5 * d' * (L_val * L_val') * d)
            V_analytical = L_val' * (E * d)
            U_analytical = L_val' * (E * d * d') * L_val - τ * I

            @test ∇̃m ≈ V_analytical rtol = 2e-2
            @test ∇̃L ≈ U_analytical rtol = 2e-2
        end

        @testset "EuclidianManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=EuclidianManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, L_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [exp(d' * z_i) * d for z_i in z]

            ∇̃m, ∇̃L = natgrad(rule, state, dx)

            E = exp(d' * m_2d + 0.5 * d' * (L_val * L_val') * d)
            ∇̃m_analytical = (L_val * L_val') * (E * d)
            Lᵀ∇L = L_val' * (E * d * d') * L_val - τ * I
            ∇̃L_analytical = L_val * (LowerTriangular(Lᵀ∇L) - 0.5 * Diagonal(Lᵀ∇L))

            @test ∇̃m ≈ ∇̃m_analytical rtol = 2e-2
            @test ∇̃L ≈ ∇̃L_analytical rtol = 2e-2
        end

        @testset "RiemannianManifold (FullNormal)" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=RiemannianManifold())
            state_init = Optimisers.init(rule, m_2d)

            state = (
                q=(m_2d, S_val),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [exp(d' * z_i) * d for z_i in z]

            ∇̃m, ∇̃S = natgrad(rule, state, dx)

            E = exp(d' * m_2d + 0.5 * d' * inv(S_val) * d)
            ∇̃m_analytical = inv(S_val) * (E * d)
            Ĝ = τ * S_val - E * d * d'
            ∇̃S_analytical = Ĝ - η / 2 * (Ĝ * inv(S_val) * Ĝ)

            @test ∇̃m ≈ ∇̃m_analytical rtol = 2e-2
            @test ∇̃S ≈ ∇̃S_analytical rtol = 2e-2
        end
    end
end
