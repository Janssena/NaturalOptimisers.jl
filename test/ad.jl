using NaturalOptimisers
using Test
using Random
using ForwardDiff
using LinearAlgebra
using Statistics

@testset "Automatic Differentiation Natural Gradient Verification" begin
    rng = Random.MersenneTwister(1234)
    N = 1_000_000  # High sample count for MC convergence
    τ = 1.0
    η = 0.1

    # Bayesian Linear Regression conjugate quadratic parameters
    A = [2.5 0.8; 0.8 1.8]
    b = [-1.2, 0.5]
    
    m_val = [0.8, -0.4]
    L_val = LowerTriangular([1.4 0.0; 0.4 0.9])
    S_val = Symmetric(inv(L_val * L_val'))

    @testset "LieGroupManifold (FullNormal) AD" begin
        rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=LieGroupManifold())
        state_init = Optimisers.init(rule, m_val)
        
        state = (
            q=(m_val, L_val),
            momentum=state_init.momentum,
            epsilon=[randn(rng, Float64, 2) for _ in 1:N]
        )

        z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
        dx = [ForwardDiff.gradient(zi -> 0.5 * zi' * A * zi + b' * zi, zi) for zi in z]

        ∇̃m, ∇̃L = natgrad(rule, state, dx)

        # ForwardDiff ELBO/Objective evaluation
        # θ = [m_1, m_2, L_11, L_21, L_22]
        function obj_lie(θ)
            m = θ[1:2]
            L = [θ[3] 0.0; θ[4] θ[5]]
            Σ = L * L'
            E = 0.5 * m' * A * m + 0.5 * tr(A * Σ) + b' * m
            H = log(θ[3]) + log(θ[5])
            return E - τ * H
        end

        θ_val = [m_val[1], m_val[2], L_val[1,1], L_val[2,1], L_val[2,2]]
        grad_θ = ForwardDiff.gradient(obj_lie, θ_val)

        g_m = grad_θ[1:2]
        g_L = [grad_θ[3] 0.0; grad_θ[4] grad_θ[5]]

        V_ref = L_val' * g_m
        U_ref = L_val' * g_L

        @test ∇̃m ≈ V_ref rtol=2e-2
        @test LowerTriangular(∇̃L) ≈ LowerTriangular(U_ref) rtol=2e-2
    end

    @testset "EuclidianManifold (FullNormal) AD" begin
        rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=EuclidianManifold())
        state_init = Optimisers.init(rule, m_val)
        
        state = (
            q=(m_val, L_val),
            momentum=state_init.momentum,
            epsilon=[randn(rng, Float64, 2) for _ in 1:N]
        )

        z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
        dx = [ForwardDiff.gradient(zi -> 0.5 * zi' * A * zi + b' * zi, zi) for zi in z]

        ∇̃m, ∇̃L = natgrad(rule, state, dx)

        function obj_euclidian(θ)
            m = θ[1:2]
            L = [θ[3] 0.0; θ[4] θ[5]]
            Σ = L * L'
            E = 0.5 * m' * A * m + 0.5 * tr(A * Σ) + b' * m
            H = log(θ[3]) + log(θ[5])
            return E - τ * H
        end

        θ_val = [m_val[1], m_val[2], L_val[1,1], L_val[2,1], L_val[2,2]]
        grad_θ = ForwardDiff.gradient(obj_euclidian, θ_val)

        g_m = grad_θ[1:2]
        g_L = [grad_θ[3] 0.0; grad_θ[4] grad_θ[5]]

        ∇̃m_ref = (L_val * L_val') * g_m
        Lᵀ∇L = L_val' * g_L
        ∇̃L_ref = L_val * (LowerTriangular(Lᵀ∇L) - 0.5 * Diagonal(Lᵀ∇L))

        @test ∇̃m ≈ ∇̃m_ref rtol=2e-2
        @test ∇̃L ≈ ∇̃L_ref rtol=2e-2
    end

    @testset "RiemannianManifold (FullNormal) AD" begin
        rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=RiemannianManifold())
        state_init = Optimisers.init(rule, m_val)
        
        state = (
            q=(m_val, S_val),
            momentum=state_init.momentum,
            epsilon=[randn(rng, Float64, 2) for _ in 1:N]
        )

        z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
        dx = [ForwardDiff.gradient(zi -> 0.5 * zi' * A * zi + b' * zi, zi) for zi in z]

        ∇̃m, ∇̃S = natgrad(rule, state, dx)

        function obj_riemannian(θ)
            m = θ[1:2]
            S = [θ[3] θ[4]; θ[4] θ[5]]
            Σ = inv(S)
            E = 0.5 * m' * A * m + 0.5 * tr(A * Σ) + b' * m
            H = -0.5 * log(det(S))
            return E - τ * H
        end

        θ_val = [m_val[1], m_val[2], S_val[1,1], S_val[2,1], S_val[2,2]]
        grad_θ = ForwardDiff.gradient(obj_riemannian, θ_val)

        g_m = grad_θ[1:2]
        # Map unrolled gradient back to symmetric matrix format
        g_S = [grad_θ[3] 0.5*grad_θ[4]; 0.5*grad_θ[4] grad_θ[5]]

        ∇̃m_ref = inv(S_val) * g_m
        Ĝ = τ * S_val - A
        ∇̃S_ref = Ĝ - η / 2 * (Ĝ * inv(S_val) * Ĝ)

        @test ∇̃m ≈ ∇̃m_ref rtol=2e-2
        @test ∇̃S ≈ ∇̃S_ref rtol=2e-2
    end

    @testset "DiagNormal AD Verification" begin
        m_diag = [0.8, -0.4]
        σ_diag = [1.2, 0.7]
        ϕ_diag = @. log(exp(σ_diag) - 1.0) # softplusinv(σ_diag)
        s_diag = 1.0 ./ (σ_diag .^ 2)      # Precision

        @testset "LieGroupManifold (DiagNormal) AD" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=LieGroupManifold())
            state_init = Optimisers.init(rule, m_diag)
            
            state = (
                q=(m_diag, σ_diag),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [ForwardDiff.gradient(zi -> 0.5 * zi' * A * zi + b' * zi, zi) for zi in z]

            ∇̃m, ∇̃σ = natgrad(rule, state, dx)

            function obj_lie_diag(θ)
                m = θ[1:2]
                σ = θ[3:4]
                E = 0.5 * m' * A * m + 0.5 * sum(diag(A) .* (σ.^2)) + b' * m
                H = sum(log.(σ))
                return E - τ * H
            end

            θ_val = [m_diag[1], m_diag[2], σ_diag[1], σ_diag[2]]
            grad_θ = ForwardDiff.gradient(obj_lie_diag, θ_val)

            g_m = grad_θ[1:2]
            g_σ = grad_θ[3:4]

            V_ref = σ_diag .* g_m
            U_ref = σ_diag .* g_σ

            @test ∇̃m ≈ V_ref rtol=2e-2
            @test ∇̃σ ≈ U_ref rtol=2e-2
        end

        @testset "EuclidianManifold (DiagNormal) AD" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=EuclidianManifold())
            state_init = Optimisers.init(rule, m_diag)
            
            state = (
                q=(m_diag, ϕ_diag),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [ForwardDiff.gradient(zi -> 0.5 * zi' * A * zi + b' * zi, zi) for zi in z]

            ∇̃m, ∇̃ϕ = natgrad(rule, state, dx)

            function obj_euclidian_diag(θ)
                m = θ[1:2]
                ϕ = θ[3:4]
                σ = log.(1.0 .+ exp.(ϕ)) # softplus(ϕ)
                E = 0.5 * m' * A * m + 0.5 * sum(diag(A) .* (σ.^2)) + b' * m
                H = sum(log.(σ))
                return E - τ * H
            end

            θ_val = [m_diag[1], m_diag[2], ϕ_diag[1], ϕ_diag[2]]
            grad_θ = ForwardDiff.gradient(obj_euclidian_diag, θ_val)

            g_m = grad_θ[1:2]
            g_ϕ = grad_θ[3:4]

            ∇̃m_ref = (σ_diag .^ 2) .* g_m
            # change of variables scale factor based on inverse F_ϕ: σ^2 / (2 * logistic(ϕ)^2)
            logistic(x) = 1.0 / (1.0 + exp(-x))
            ∇̃ϕ_ref = (σ_diag .^ 2) ./ (2.0 .* (logistic.(ϕ_diag) .^ 2)) .* g_ϕ

            @test ∇̃m ≈ ∇̃m_ref rtol=2e-2
            @test ∇̃ϕ ≈ ∇̃ϕ_ref rtol=2e-2
        end

        @testset "RiemannianManifold (DiagNormal) AD" begin
            rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=true, manifold=RiemannianManifold())
            state_init = Optimisers.init(rule, m_diag)
            
            state = (
                q=(m_diag, s_diag),
                momentum=state_init.momentum,
                epsilon=[randn(rng, Float64, 2) for _ in 1:N]
            )

            z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
            dx = [ForwardDiff.gradient(zi -> 0.5 * zi' * A * zi + b' * zi, zi) for zi in z]

            ∇̃m, ∇̃s = natgrad(rule, state, dx)

            function obj_riemannian_diag(θ)
                m = θ[1:2]
                s = θ[3:4]
                E = 0.5 * m' * A * m + 0.5 * sum(diag(A) ./ s) + b' * m
                H = -0.5 * sum(log.(s))
                return E - τ * H
            end

            θ_val = [m_diag[1], m_diag[2], s_diag[1], s_diag[2]]
            grad_θ = ForwardDiff.gradient(obj_riemannian_diag, θ_val)

            g_m = grad_θ[1:2]

            ∇̃m_ref = (1.0 ./ s_diag) .* g_m
            G = τ .* s_diag .- diag(A)
            ∇̃s_ref = G .- (η / 2.0) .* G .* (1.0 ./ s_diag) .* G

            @test ∇̃m ≈ ∇̃m_ref rtol=2e-2
            @test ∇̃s ≈ ∇̃s_ref rtol=2e-2
        end
    end

    @testset "Monte Carlo Fisher Information Matrix (MC FIM) Verification" begin
        # 2D Gaussian LieGroup parameterization: θ = [m_1, m_2, L_11, L_21, L_22]
        θ_val = [m_val[1], m_val[2], L_val[1,1], L_val[2,1], L_val[2,2]]

        function log_q(θ, z)
            m = θ[1:2]
            L = [θ[3] 0.0; θ[4] θ[5]]
            Σ = L * L'
            diff = z - m
            return -0.5 * diff' * inv(Σ) * diff - 0.5 * log(det(Σ)) - log(2π)
        end

        function analytical_fim(θ)
            m(t) = t[1:2]
            Σ(t) = [t[3] 0.0; t[4] t[5]] * [t[3] 0.0; t[4] t[5]]'
            
            J_m = ForwardDiff.jacobian(m, θ)
            Σ_val = Σ(θ)
            inv_Σ = inv(Σ_val)
            
            unroll_Σ(t) = vec(Σ(t))
            J_Σ = ForwardDiff.jacobian(unroll_Σ, θ)
            
            F = zeros(5, 5)
            for j in 1:5, k in 1:5
                term_mean = J_m[:, j]' * inv_Σ * J_m[:, k]
                dΣ_dj = reshape(J_Σ[:, j], 2, 2)
                dΣ_dk = reshape(J_Σ[:, k], 2, 2)
                term_cov = 0.5 * tr(inv_Σ * dΣ_dj * inv_Σ * dΣ_dk)
                F[j, k] = term_mean + term_cov
            end
            return F
        end

        # Sample for MC FIM estimation
        N_mc = 20_000
        z_samples = [L_val * randn(rng, Float64, 2) + m_val for _ in 1:N_mc]
        scores = [ForwardDiff.gradient(t -> log_q(t, zi), θ_val) for zi in z_samples]
        
        F_MC = mean(s * s' for s in scores)
        F_exact = analytical_fim(θ_val)

        @test F_MC ≈ F_exact rtol=5e-2
    end

    @testset "EuclidianManifold (FullNormal) Natural Gradients via MC FIM" begin
        # 2D Gaussian Euclidean parameterization: θ = [m_1, m_2, L_11, L_21, L_22]
        θ_val = [m_val[1], m_val[2], L_val[1,1], L_val[2,1], L_val[2,2]]

        function log_q(θ, z)
            m = θ[1:2]
            L = [θ[3] 0.0; θ[4] θ[5]]
            Σ = L * L'
            diff = z - m
            return -0.5 * diff' * inv(Σ) * diff - 0.5 * log(det(Σ)) - log(2π)
        end

        function obj_euclidian(θ)
            m = θ[1:2]
            L = [θ[3] 0.0; θ[4] θ[5]]
            Σ = L * L'
            E = 0.5 * m' * A * m + 0.5 * tr(A * Σ) + b' * m
            H = log(θ[3]) + log(θ[5])
            return E - τ * H
        end

        # 1. Compute objective gradient via ForwardDiff
        g_θ = ForwardDiff.gradient(obj_euclidian, θ_val)

        # 2. Compute F_MC
        N_mc = 50_000
        z_samples = [L_val * randn(rng, Float64, 2) + m_val for _ in 1:N_mc]
        scores = [ForwardDiff.gradient(t -> log_q(t, zi), θ_val) for zi in z_samples]
        F_MC = mean(s * s' for s in scores)

        # 3. Reference natural gradient via inv(F_MC) * g_θ
        nat_g_θ = inv(F_MC) * g_θ

        nat_m_ref = nat_g_θ[1:2]
        nat_L_ref = LowerTriangular([nat_g_θ[3] 0.0; nat_g_θ[4] nat_g_θ[5]])

        # 4. Our library's natural gradients
        rule = NaturalDescent(η; tau=τ, scale=1.5, meanfield=false, manifold=EuclidianManifold())
        state_init = Optimisers.init(rule, m_val)
        state = (
            q=(m_val, L_val),
            momentum=state_init.momentum,
            epsilon=[randn(rng, Float64, 2) for _ in 1:N]
        )
        z = [NaturalOptimisers.sample(rule, state, i) for i in 1:N]
        dx = [ForwardDiff.gradient(zi -> 0.5 * zi' * A * zi + b' * zi, zi) for zi in z]
        ∇̃m, ∇̃L = natgrad(rule, state, dx)

        @test ∇̃m ≈ nat_m_ref rtol=2e-2
        @test ∇̃L ≈ nat_L_ref rtol=2e-2
    end
end
