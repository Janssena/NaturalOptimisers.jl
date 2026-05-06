import Optimisers: @..

struct NaturalDescent{Q<:Distribution,M<:AbstractManifold,T} <: Optimisers.AbstractRule
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
    momentum=(initq(o, x; scale=zero(T))..., o.beta), # mt, vt, βt
    epsilon=[zero(x)] # Should be a vector of samples
)

Optimisers.apply!(o::NaturalDescent, state, x::AbstractArray{T}, dx::AbstractArray{T}) where T =
    Optimisers.apply!(o, state, x, [dx])

function Optimisers.apply!(o::NaturalDescent, state, ::AbstractArray{T}, dx::AbstractVector{<:AbstractArray{T}}) where T<:Real
    β = T.(o.beta)
    dmt, dvt, βt = state.momentum

    ∇̃m, ∇̃v = natgrad(o, state, dx)

    # Momentum should be using undamped velocities:
    @.. dmt = β[1] * dmt + (1 - β[1]) * ∇̃m
    @.. dvt = β[2] * dvt + (1 - β[2]) * ∇̃v

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

until_leafs(x) = Functors.isleaf(x)
until_leafs(x::Optimisers.Leaf) = true

function update_epsilon!(rng, tree::NamedTuple; num_samples::Int=1)
    fmap(tree; exclude=until_leafs) do leaf
        update_epsilon!(rng, leaf; num_samples)
    end

    return nothing
end

update_epsilon!(rng, o::Optimisers.Leaf; kwargs...) = nothing

function update_epsilon!(rng, o::Optimisers.Leaf{<:NaturalDescent}; num_samples::Int=1)
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

    ps_new = map(1:num_samples) do m
        return fmap_with_path(ps; until_leafs) do (kp, x)
            leaf = getkeypath(tree, kp)
            if leaf.rule isa NaturalDescent
                return sample(leaf.rule, leaf.state, m)
            else
                return x
            end
        end
    end

    return isone(num_samples) ? only(ps_new) : ps_new
end

Optimisers.update(tree, ps, grads::Vararg; kwargs...) =
    Optimisers.update(tree, ps, merge_grads(tree, grads...); kwargs...)

merge_grads(tree, grads::Vararg) =
    fmap_with_path(grads...) do (kp, gs)
        leaf = getkeypath(tree, kp)
        if leaf.rule isa NaturalDescent
            return [gs...]
        else
            filtered_gs = filter(!isnothing, gs)
            return isempty(gs) ? nothing : mean(filtered_gs)
        end
    end
