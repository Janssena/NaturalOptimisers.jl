module NaturalOptimisers

import Random
import Optimisers

using Functors, LinearAlgebra, Distributions, LogExpFunctions

"""
    AbstractManifold

Supertype for the geometries a [`NaturalDescent`](@ref) rule can use to
parameterise and retract the scale/covariance of its Gaussian posterior:
[`LieGroupManifold`](@ref), [`RiemannianManifold`](@ref), and
[`EuclidianManifold`](@ref). Each pairs a [`Parameterisation`](@ref) with a
`natgrad`/`update` pair that keeps the covariance positive-definite.
"""
abstract type AbstractManifold end

"""
    AbstractNaturalRule <: Optimisers.AbstractRule

Common supertype for every rule in this package that stores a Gaussian
variational distribution `q` inside the optimiser state and supports
[`sample`](@ref). [`NaturalDescent`](@ref), [`IVON`](@ref), and [`EVON`](@ref)
subtype it, so they share the [`sample`](@ref)/[`update_epsilon!`](@ref) and
multi-sample tree-`update` machinery defined in `rules.jl`. In a `sample` walk,
leaves whose rule is an `AbstractNaturalRule` yield a posterior draw; all other
leaves keep their point estimate.
"""
abstract type AbstractNaturalRule <: Optimisers.AbstractRule end

# Optimiser rules are immutable configuration, not parameter containers, so treat them as
# Functors leaves. This stops tree walks (e.g. the `fmap(copy, …)` inside `Optimisers.update`)
# from deconstructing and rebuilding a rule via `ConstructionBase` — which fails for
# `NaturalDescent` because its variational-family parameter `Q` is a phantom type parameter
# (set from `meanfield`, carried by no field) that cannot be recovered from the fields.
Functors.@leaf AbstractNaturalRule

include("lib/rules.jl")
# TODO: we should probably NOT export update_epsilon!, we want users to use the sample function.
# Using update_epsilon! incorrectly breaks the link between the ps sample and the actual epsilon 
# in the rule.
export NaturalDescent, update_epsilon!, sample

include("lib/manifolds.jl")
export AbstractManifold, RiemannianManifold, LieGroupManifold, EuclidianManifold, initq

include("lib/lie_groups.jl")
include("lib/riemannian.jl")
include("lib/euclidian.jl")
export natgrad, update

include("lib/evon.jl")
export EVON

include("lib/ivon.jl")
export IVON

end # module NaturalOptimisers
