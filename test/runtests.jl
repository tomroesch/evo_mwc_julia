include("../src/gaussianprocesses.jl")
using .gaussianprocesses, Test

@testset "Type structure" begin
    @test lnGP <: AbstractGaussianProcess
    @test nnGP <: AbstractGaussianProcess
    @test sqexpGP <: AbstractGaussianProcess
    @test sqexplinGP <: AbstractGaussianProcess
    @test maternGP <: AbstractGaussianProcess
end
