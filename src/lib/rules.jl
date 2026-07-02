import Optimisers: @..

struct NaturalDescent{Q<:Distribution,M<:AbstractManifold,T} <: AbstractNaturalRule
    eta::T # learning rate
    beta::NTuple{2,T} # Momentum parameters
    tau::T # Temperature parameter (1 = pure Bayesian Inference)
    init_scale::T
    manifold::M
end

function NaturalDescent(eta, beta=(0.9, 0.99); tau=1, scale=1, meanfield::Bool=true, manifold::AbstractManifold=LieGroupManifold())
    (tau < 0 || tau > 1) && throw(ErrorException("`tau` must be between 0 (standard optimisation) and 1 (pure Bayesian Inference)."))

    Q = meanfield ? DiagNormal : FullNormal
    T = eltype(eta)
    return NaturalDescent{Q,typeof(manifold),T}(eta, T.(beta), T(tau), T(scale), manifold)
end

Optimisers.init(o::NaturalDescent{Q}, x::AbstractVector{T}) where {Q,T} = (
    q=initq(o, x; scale=T(o.init_scale)), # parameters of variational distribution
    # Momentum (mt, vt) starts at zero, matching the structure/type of `q`. Note we
    # zero out the `q` template rather than calling `initq(...; scale=0)`: the latter
    # would give 1/scale = Inf for the precision parameterisation (RiemannianManifold).
    momentum=(map(zero, initq(o, x; scale=T(o.init_scale)))..., o.beta), # mt, vt, βt
    epsilon=[zero(x)] # Should be a vector of samples
)

Optimisers.apply!(o::NaturalDescent, state, x::AbstractArray{T}, dx::AbstractArray{T}) where T =
    Optimisers.apply!(o, state, x, [dx])

function Optimisers.apply!(o::NaturalDescent, state, ::AbstractArray{T}, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    β = T.(o.beta)
    dmt, dvt, βt = state.momentum

    ∇̃m, ∇̃v = natgrad(o, state, dx)

    # Momentum uses undamped velocities. We rebuild (rather than mutate in place) and re-wrap
    # each EMA to the structural type of the previous momentum (`_match_momentum`). This keeps
    # the `Optimisers.Leaf`'s state field type stable across steps: the natural-gradient scale
    # velocity `∇̃v` is a dense matrix, but for a `FullNormal` the momentum slot is initialised
    # as the q-template's `LowerTriangular`/`Symmetric` scale. Without re-wrapping, the
    # single-gradient `Optimisers.update` path (which assigns into the *mutable* Leaf via
    # `setproperty!`) would try to `convert` a `Matrix` into that wrapper and throw.
    dmt = _match_momentum(dmt, @. β[1] * dmt + (1 - β[1]) * ∇̃m)
    dvt = _match_momentum(dvt, @. β[2] * dvt + (1 - β[2]) * ∇̃v)

    # bias-corrected versions for the update
    mt_hat = dmt ./ (1 .- βt[1])
    vt_hat = dvt ./ (1 .- βt[2])

    state′ = (
        q=update(o, state.q..., mt_hat, vt_hat),
        momentum=(dmt, dvt, βt .* β),
        epsilon=state.epsilon
    )

    return state′, nothing
end

# Re-wrap a freshly computed momentum array into the structural type of the previous momentum
# `template`, so the momentum keeps a stable type across steps (see `apply!`). A `FullNormal`
# scale momentum is a `LowerTriangular` (Lie/Euclidian) or `Symmetric` (Riemannian) template;
# everything else (diagonal `Vector`s, plain matrices) is returned unchanged.
_match_momentum(::T, x) where {T<:AbstractArray} = x
_match_momentum(::LowerTriangular, x) = LowerTriangular(x)
_match_momentum(::Symmetric, x) = Symmetric(x)

until_leafs(x) = Functors.isleaf(x)
until_leafs(x::Optimisers.Leaf) = true

# Generic over the container (NamedTuple / Tuple / Vector / nested combinations); the
# `Optimisers.Leaf` methods below are more specific and handle the actual leaves.
function update_epsilon!(rng, tree; num_samples::Int=1)
    fmap(tree; exclude=until_leafs) do leaf
        update_epsilon!(rng, leaf; num_samples)
    end

    return nothing
end

update_epsilon!(rng, o::Optimisers.Leaf; kwargs...) = nothing

function update_epsilon!(rng, o::Optimisers.Leaf{<:AbstractNaturalRule}; num_samples::Int=1)
    dims = size(first(o.state.epsilon))
    T = eltype(first(o.state.epsilon))

    empty!(o.state.epsilon)
    for _ in 1:num_samples
        push!(o.state.epsilon, randn(rng, T, dims))
    end

    return nothing
end

function sample(rng, ps, tree; num_samples::Int=1)
    update_epsilon!(rng, tree; num_samples)

    # Walk the parameters `ps` and the optimiser-state `tree` in parallel: at each
    # parameter leaf `x` the matching `tree` node is its `Optimisers.Leaf`. Leaves
    # whose rule is an `AbstractNaturalRule` are replaced by a posterior sample;
    # all others (e.g. an Adam leaf) keep their point estimate `x`.
    ps_new = map(1:num_samples) do m
        return fmap(ps, tree; exclude=until_leafs) do x, leaf
            if leaf.rule isa AbstractNaturalRule
                return sample(leaf.rule, leaf.state, m)
            else
                return x
            end
        end
    end

    return isone(num_samples) ? only(ps_new) : ps_new
end

# Multi-sample tree update: `grads` is a vector of per-sample gradient trees (as returned
# by `sample(rng, ps, tree; num_samples)` paired with a loss gradient). Each leaf's rule is
# applied once with all of its per-sample gradients, so a variational rule receives the full
# Monte-Carlo batch (its `apply!` averages internally) while ordinary rules get the mean.
#
# We do the walk ourselves rather than delegate to `Optimisers.update`, because Optimisers'
# higher-order method `update(tree, model, grad, higher...)` treats extra gradients as
# higher-order terms and keeps only the first (`apply!(o, state, x, dx, dxs...) = ... dx`).
# Dispatching on a `Vector` of structured gradients avoids clashing with the single-gradient
# `Optimisers.update(tree, model, grad)` (a lone array gradient is not such a vector).
function Optimisers.update(tree, ps, grads::AbstractVector{<:Union{AbstractArray,Tuple,NamedTuple}})
    # Copy first (as `Optimisers.update` does) so in-place rule updates don't touch the caller's
    # arrays — both the state tree and the parameters may be written to below.
    tree = fmap(copy, tree; exclude=Optimisers.maywrite)
    ps = fmap(copy, ps; exclude=Optimisers.maywrite)
    paired = fmap(tree, ps, grads...; exclude=until_leafs) do leaf, x, gs...
        _update_leaf(leaf, x, gs)
    end
    new_tree = fmap(p -> p[1], paired; exclude=_is_leaf_pair)
    new_ps = fmap(p -> p[2], paired; exclude=_is_leaf_pair)
    return new_tree, new_ps
end

_is_leaf_pair(x) = x isa Tuple{<:Optimisers.Leaf,<:Any}

# Apply a single leaf's rule to its per-sample gradient batch `gs`, returning the
# (updated Leaf, updated parameter). Variational leaves leave the parameter untouched
# (their state holds the posterior); ordinary leaves are updated with the mean gradient.
function _update_leaf(leaf::Optimisers.Leaf, x, gs)
    leaf.frozen && return (leaf, x)
    if leaf.rule isa AbstractNaturalRule
        state, _ = Optimisers.apply!(leaf.rule, leaf.state, x, collect(gs))
        return (Optimisers.Leaf(leaf.rule, state; frozen=leaf.frozen), x)
    else
        valid = filter(!isnothing, gs)
        isempty(valid) && return (leaf, x)
        state, Δ = Optimisers.apply!(leaf.rule, leaf.state, x, mean(valid))
        return (Optimisers.Leaf(leaf.rule, state; frozen=leaf.frozen), Optimisers.subtract!(x, Δ))
    end
end

# A model that is itself a bare vector of arrays is ambiguous: a single gradient for it is a
# `Vector{<:AbstractArray}`, indistinguishable from the per-sample gradient batch consumed by the
# multi-sample tree `update`. Reject it at setup time with an actionable message.
_vector_model_error() = throw(ErrorException(
    "NaturalOptimisers: a model that is a bare `Vector` of arrays is ambiguous with the per-sample " *
    "gradient batch used by the tree-level `Optimisers.update`. Wrap the model in a `NamedTuple` or `Tuple`."))

Optimisers.setup(::AbstractNaturalRule, ::AbstractVector{<:AbstractArray}) = _vector_model_error()
