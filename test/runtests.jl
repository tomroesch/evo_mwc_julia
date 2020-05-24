include("../src/gaussianprocesses.jl")
using .gaussianprocesses, Test, LinearAlgebra, BenchmarkTools

lthbounds = Float64[-2 2; -2 2; -4 -2]
x = collect(1.:3.)
y = [1., 3., 1.2]
lth = [0, 0, -3]

@testset "Type structure" begin
    @testset "highest level types" begin
        @test_broken lnGP <: AbstractGaussianProcess
        @test_broken nnGP <: AbstractGaussianProcess
        @test sqexpGP <: AbstractGaussianProcess
        @test_broken sqexplinGP <: AbstractGaussianProcess
        @test_broken maternGP <: AbstractGaussianProcess
    end
    @testset "Infer Field types" begin
        g1 = sqexpGP(lthbounds, x, y)
        @test typeof(g1) == sqexpGP{Array{Float64, 2},Array{Float64,1},Array{Float64,1},Array{Float64,1},Array{Float64,1}}
    end
end

g = sqexpGP(lthbounds, x, y)
# An error here might come from weird behavior of ≈
@testset "squared exponential covariance function" begin
    out = gaussianprocesses.covfn(g, x[1], x , lth)
    @test out[1] ≈ [1., 0.60653066, 0.13533528]
    @test out[2] ≈ [ 1. -0.;
                     0.60653066 -0.30326533;
                     0.13533528 -0.27067057]
end


@testset "squared exponential covariance function kernel matrix" begin
    out = gaussianprocesses.kernelmatrix(g, lth, x)
    @test out[1] ≈  [1. 0.60653066 0.13533528;
                     0.60653066 1. 0.60653066;
                     0.13533528 0.60653066 1.]
    #@test UpperTriangular(out[2]) ≈ UpperTriangular([1.02459117 0.59197334 0.13208711;
                                                    # 0.60653066 0.83627426 0.63177673;
                                                    # 0.13533528 0.60653066 0.79573754])
end


@testset "negative of log marginal likelihood" begin
    @test gaussianprocesses.nlml(g, lth) ≈ 7.49954813185127
    @test map(x->gaussianprocesses.nlml(g, x), [lth])[1] ≈ 7.49954813185127
end


@testset "Jacobian of negative log marginal likelihood" begin
    @test gaussianprocesses.jacnlml(g, lth) ≈ [-3.20149247, -2.41355255, -0.42423092]
end


@testset "Optimization" begin
    @test gaussianprocesses.findhyperparameters(g, noruns=5) ≈ ([1.1647105451250055, 0.3196261652824192, -15.905694493274043], 5.680029993560606)
end
