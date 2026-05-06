module NaturalOptimisers

import Random
import Optimisers

using Functors, LinearAlgebra, Distributions, LogExpFunctions

abstract type AbstractManifold end

include("lib/rules.jl")
export NaturalDescent, update_epsilon!, sample

include("lib/manifolds.jl")
export AbstractManifold, RiemannianManifold, LieGroupManifold, EuclidianManifold, initq

include("lib/lie_groups.jl")
include("lib/riemannian.jl")
include("lib/euclidian.jl")
export natgrad, update

end # module NaturalOptimisers
