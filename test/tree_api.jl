using NaturalOptimisers
using Test
using Random
using Statistics
using Optimisers
using LinearAlgebra
using LogExpFunctions

# Train a model through the shared tree API (setup → sample → update) for `nsteps`,
# drawing `nsamp` Monte-Carlo samples per step and forming per-sample gradients with
# `gradfn(sample) -> gradient_tree`. Returns the final optimiser-state tree.
function train_tree(rng, model, tree, gradfn; nsteps, nsamp)
    for _ in 1:nsteps
        s = sample(rng, model, tree; num_samples=nsamp)
        g = map(gradfn, s)
        tree, _ = Optimisers.update(tree, model, g)
    end
    return tree
end

@testset "Tree-level API across rules and containers" begin

    @testset "NaturalDescent through Optimisers.update" begin
        # Regression guard for `Functors.@leaf`: NaturalDescent's phantom variational-family
        # parameter `Q` previously made the rule impossible to reconstruct during the copy inside
        # `Optimisers.update`, so the whole tree API (standard and multi-sample) errored for it.
        rng = MersenneTwister(101)
        a = [3.0, 1.0, 0.5, 2.0]
        b = randn(rng, 4)
        model = (w=zeros(4),)
        gradfn = x -> (w=a .* x.w .+ b,)
        for man in (LieGroupManifold(), RiemannianManifold(), EuclidianManifold())
            @testset "$(nameof(typeof(man)))" begin
                rule = NaturalDescent(0.01; tau=1.0, scale=1.0, meanfield=true, manifold=man)
                tree = Optimisers.setup(rule, model)
                tree = train_tree(MersenneTwister(7), model, tree, gradfn; nsteps=8000, nsamp=32)
                σ² = man isa LieGroupManifold ? tree.w.state.q[2] .^ 2 :
                     man isa RiemannianManifold ? 1 ./ tree.w.state.q[2] :
                     softplus.(tree.w.state.q[2]) .^ 2
                @test tree.w.state.q[1] ≈ -b ./ a rtol = 1e-1
                @test σ² ≈ 1 ./ a rtol = 5e-2
            end
        end
    end

    @testset "FullNormal NaturalDescent through Optimisers.update (both paths)" begin
        # Full-covariance NaturalDescent must recover the *full* Gaussian posterior through the
        # tree API, on BOTH the multi-sample vector `update` and the single-gradient `update`.
        # The single-gradient path assigns into the mutable `Optimisers.Leaf` via `setproperty!`,
        # which previously threw `convert(LowerTriangular, ::Matrix)` because the scale momentum
        # (a dense natural gradient) no longer matched the q-template wrapper frozen at init.
        # Target: ℓ(w)=½wᵀAw+bᵀw (folded prior) ⇒ posterior N(-A⁻¹b, A⁻¹).
        A = [2.0 0.3; 0.3 1.5]
        bb = [0.5, -0.2]
        true_mean = -A \ bb
        true_cov = inv(A)
        model = (w=zeros(2),)
        gradfn = x -> (w=A * x.w .+ bb,)
        # Σ from the stored scale: Riemannian holds precision S (Σ=S⁻¹); Lie/Euclidian hold a
        # Cholesky factor L (Σ=LLᵀ).
        cov_of(sc, man) = man isa RiemannianManifold ? inv(Matrix(sc)) : (L = Matrix(sc); L * L')

        for man in (LieGroupManifold(), RiemannianManifold(), EuclidianManifold())
            @testset "$(nameof(typeof(man)))" begin
                # Multi-sample vector path.
                rule = NaturalDescent(0.002; tau=1.0, scale=1.0, meanfield=false, manifold=man)
                tree = Optimisers.setup(rule, model)
                tree = train_tree(MersenneTwister(7), model, tree, gradfn; nsteps=12000, nsamp=32)
                @test tree.w.state.q[1] ≈ true_mean rtol = 5e-2
                @test cov_of(tree.w.state.q[2], man) ≈ true_cov rtol = 5e-2

                # Single-gradient path (one sample per step, scalar gradient tree).
                rng = MersenneTwister(11)
                tree = Optimisers.setup(rule, model)
                for _ in 1:20000
                    s = sample(rng, model, tree)
                    tree, _ = Optimisers.update(tree, model, gradfn(s))
                end
                @test tree.w.state.q[1] ≈ true_mean rtol = 1e-1
                @test cov_of(tree.w.state.q[2], man) ≈ true_cov rtol = 1e-1
            end
        end
    end

    @testset "Nested NamedTuple model (IVON)" begin
        rng = MersenneTwister(102)
        aW = [2.0, 1.0, 0.5]; bW = randn(rng, 3)
        ah = [3.0, 0.8]; bh = randn(rng, 2)
        model = (enc=(w=zeros(3),), dec=(w=zeros(2),))
        gradfn = x -> (enc=(w=aW .* x.enc.w .+ bW,), dec=(w=ah .* x.dec.w .+ bh,))
        rule = IVON(0.2, (0.9, 0.99); delta=1.0, lambda=1.0, init_scale=1.0)
        tree = Optimisers.setup(rule, model)
        tree = train_tree(MersenneTwister(7), model, tree, gradfn; nsteps=3000, nsamp=32)
        @test tree.enc.w.state.q[1] ≈ -bW ./ (aW .+ 1) rtol = 1e-1
        @test tree.dec.w.state.q[1] ≈ -bh ./ (ah .+ 1) rtol = 1e-1
    end

    @testset "Tuple container model (IVON)" begin
        rng = MersenneTwister(103)
        a1 = [2.0, 1.0]; b1 = randn(rng, 2)
        a2 = [3.0, 0.5, 1.0]; b2 = randn(rng, 3)
        model = (zeros(2), zeros(3))
        gradfn = x -> (a1 .* x[1] .+ b1, a2 .* x[2] .+ b2)
        rule = IVON(0.2, (0.9, 0.99); delta=1.0, lambda=1.0, init_scale=1.0)
        tree = Optimisers.setup(rule, model)
        tree = train_tree(MersenneTwister(7), model, tree, gradfn; nsteps=3000, nsamp=32)
        @test tree[1].state.q[1] ≈ -b1 ./ (a1 .+ 1) rtol = 1e-1
        @test tree[2].state.q[1] ≈ -b2 ./ (a2 .+ 1) rtol = 1e-1
    end

    @testset "Mixed rules in one tree (EVON 2D + IVON 1D + Adam 1D)" begin
        rng = MersenneTwister(104)
        model = (W=randn(rng, 4, 3), h=randn(rng, 5), b=randn(rng, 2))
        tree = (
            W=Optimisers.setup(EVON(0.1; delta=1.0, zeta=1.0), model.W),
            h=Optimisers.setup(IVON(0.1; delta=1.0, lambda=1.0), model.h),
            b=Optimisers.setup(Optimisers.Adam(0.1), model.b)
        )

        s = sample(MersenneTwister(7), model, tree; num_samples=4)
        @test size(s[1].W) == (4, 3) && length(s[1].h) == 5
        @test s[1].b == model.b                       # Adam leaf: point estimate

        g = map(x -> (W=x.W, h=x.h, b=x.b), s)
        tree2, model2 = Optimisers.update(tree, model, g)
        @test tree2.W.state.t == 1                    # EVON advanced one step
        @test tree2.h.state.q[1] != zero(model.h)     # IVON mean moved off its zero init
        @test model2.W == model.W && model2.h == model.h  # variational params stay in state
        @test model2.b != model.b                     # Adam updated the 1D leaf
    end

    @testset "Bare Vector-of-arrays model is rejected at setup" begin
        # A model that is itself a `Vector` of arrays is ambiguous with the per-sample gradient
        # batch, so `setup` errors and asks the user to wrap it in a NamedTuple/Tuple.
        @test_throws ErrorException Optimisers.setup(IVON(0.1), [zeros(3), zeros(2)])
        @test_throws ErrorException Optimisers.setup(EVON(0.1), [zeros(2, 2), zeros(2, 2)])
        @test_throws ErrorException Optimisers.setup(NaturalDescent(0.1), [zeros(3), zeros(2)])
        @test_throws ErrorException Optimisers.setup(EVON(0.1), Optimisers.Adam(0.1), [zeros(2, 2), zeros(3)])

        # Legitimate model shapes are unaffected.
        @test Optimisers.setup(IVON(0.1), zeros(4)) isa Optimisers.Leaf      # single array param
        @test Optimisers.setup(IVON(0.1), (a=zeros(3),)) isa NamedTuple
        @test Optimisers.setup(IVON(0.1), (zeros(3), zeros(2))) isa Tuple
    end
end
