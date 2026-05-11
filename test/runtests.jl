using NaturalOptimisers
using Test
using Optimisers
using LinearAlgebra
using Statistics

@testset "NaturalOptimisers.jl" begin
    include("verification.jl")
    include("ad.jl")
end
