module gaussianprocesses

using LinearAlgebra

export AbstractGaussianProcess, sqexpGP, covfn

abstract type AbstractGaussianProcess end



"""
    struct sqexpGP <: AbstractGaussianProcess

Creates a Gaussian process with squared exponential covariance function.

#Arguments
- `lthbounds` : a dictionary of pairs of the bounds on the hyperparameters in log10 space, such as Dict(0=> [0,6], 1=> [-3,4], 2=> [-6,-4])
- `x` : a 1-d array of the abscissa data
- `y` : a multi-dimensional array of the ordinate data
- `merrors` : if specified, a 1-d array of the measurement errors (as variances) v, else `nothing`
- `noparams` : Number of parameters
"""
mutable struct sqexpGP{
    T_b<:AbstractVector,
    T_x<:AbstractVector,
    T_y<:AbstractArray,
    T_xnew<:AbstractVector,
    T_merror<:AbstractVector
    } <: AbstractGaussianProcess
    b::T_b
    x::T_x
    y::T_y
    xnew::T_xnew
    merrors::T_merror
    noparams::Int
    description::String
end

sqexpGP(lthbounds::Dict, x, y) = sqexpGP([lthbounds[a] for a in keys(lthbounds)], x, y, x, Float64[], 2, "squared exponential Gaussian process")
sqexpGP(lthbounds::Dict, x, y, merrors) = sqexpGP([lthbounds[a] for a in keys(lthbounds)], x, y, x, merrors, 2, "squared exponential Gaussian process")


function info(self::sqexpGP)
    println("hparam[1] determines the amplitude of variation.")
    println("hparam[2] determines the flexibility.")
    println("hparam[3] determines the variance of the measurement error.")
end


"""
    covfn(self::sqexpGP, x, xp, lth)

Squared exponential covariance function.
Returns the kernel function and Jacobian of the kernel function.

#Arguments
- `self` : gaussianprocess with squared exponential covariance function
- `x` : a 1-d array of abscissa
- `xp` : a 1-d array of alternative abscissa
- `lth` : the log of the hyperparameters
"""
function covfn(self::sqexpGP, x, xp, lth)
    th = exp.(lth)
    e = exp.(-th[2] / 2.0 * (x .- xp).^2)
    k = th[1] * e
    jk = zeros(length(xp), self.noparams)
    jk[:, 1] = e * th[1]
    jk[:, 2] = -th[1] * th[2] * e / 2.0 .* (x .- xp).^2
    return k, jk
end


"""
    function kernelmatrix(self::sqexpGP, lth, x)

Returns kernel matrix K(X,X) supplemented with measurement noise and its Cholesky decomposition.

#Arguments
- `self` : gaussianprocess with squared exponential covariance function
- `lth` : log of the hyperparameters
- `x` : abscissa values
"""
function kernelmatrix(self::sqexpGP, lth, x)
    k = zeros(length(x), length(x))
    for i in 1:length(x)
        k[i,:]= covfn(self, x[i], x, lth)[1]
    end

    if ~isempty(self.merrors)
        kn = k .+ exp(lth[end]) * Diagonal(self.merrors)
    else
        kn = k .+ exp(lth[end]) * I(length(x))
    end

    L = cholesky(kn)
    return k, L.U
end

end  # module
Main.gaussianprocesses
