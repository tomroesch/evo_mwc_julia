include("../src/gaussianprocesses.jl")
using .gaussianprocesses, Test, LinearAlgebra

@testset "Type structure" begin
    @testset "highest level types" begin
        @test_broken lnGP <: AbstractGaussianProcess
        @test_broken nnGP <: AbstractGaussianProcess
        @test sqexpGP <: AbstractGaussianProcess
        @test_broken sqexplinGP <: AbstractGaussianProcess
        @test_broken maternGP <: AbstractGaussianProcess
    end
    @testset "Infer Field types" begin
        g1 = sqexpGP(Dict(1=> (-2,2), 2=> (-2,2)), rand(10), rand(10))
        @test typeof(g1) == sqexpGP{Array{Tuple{Int64,Int64},1},Array{Float64,1},Array{Float64,1},Array{Float64,1},Array{Float64,1}}
        g2 = sqexpGP(Dict(1=> (-2.,2.), 2=> (-2.,2.)), [1, 2], [1, 2])
        @test typeof(g2) == sqexpGP{Array{Tuple{Float64,Float64},1},Array{Int64,1},Array{Int64,1},Array{Int64,1},Array{Float64,1}}
    end
end

g = sqexpGP(Dict(1=> (-2,2), 2=> (-2,2)), [1, 2], [1, 2])

@testset "squared exponential covariance function" begin
    out = gaussianprocesses.covfn(g, [1], [1, 2] , [0, 0])
    @test out[1] == [1, exp(-1/2)]
    @test out[2] == [1. 0; exp(-1/2) (-1/2*exp(-1 / 2))]
end


@testset "squared exponential covariance function kernel matrix" begin
    out = gaussianprocesses.kernelmatrix(g, [0, 0], [1, 2])
    @test out[1] ==  [1 exp(-1/2); exp(-1/2) 1]
    @test out[2] == cholesky([1+1 exp(-1/2); exp(-1/2) 1+1]).U
end
