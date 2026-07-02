using NaturalOptimisers
using Test
using Optimisers
using LinearAlgebra
using Statistics

@testset "NaturalOptimisers.jl" begin
    include("verification.jl")
    include("ad.jl")
    include("natural_descent.jl")
    include("evon.jl")
    include("ivon.jl")
    include("logistic_regression.jl")
    include("tree_api.jl")
end
