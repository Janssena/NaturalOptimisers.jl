module NaturalOptimisers

import Random
import Optimisers

using Functors, LinearAlgebra, Distributions, LogExpFunctions

abstract type AbstractManifold end

# Common supertype for all rules in this package that store a variational
# distribution `q` inside the optimiser state and support `sample`. Both the
# manifold-based `NaturalDescent` and the eigenspace `EVON` rule subtype this,
# allowing them to share the `sample`/`update_epsilon!`/tree-`update` machinery
# defined in `rules.jl`.
abstract type AbstractNaturalRule <: Optimisers.AbstractRule end

# Optimiser rules are immutable configuration, not parameter containers, so treat them as
# Functors leaves. This stops tree walks (e.g. the `fmap(copy, …)` inside `Optimisers.update`)
# from deconstructing and rebuilding a rule via `ConstructionBase` — which fails for
# `NaturalDescent` because its variational-family parameter `Q` is a phantom type parameter
# (set from `meanfield`, carried by no field) that cannot be recovered from the fields.
Functors.@leaf AbstractNaturalRule

include("lib/rules.jl")
export NaturalDescent, update_epsilon!, sample

include("lib/manifolds.jl")
export AbstractManifold, RiemannianManifold, LieGroupManifold, EuclidianManifold, initq

include("lib/lie_groups.jl")
include("lib/riemannian.jl")
include("lib/euclidian.jl")
export natgrad, update

end # module NaturalOptimisers
