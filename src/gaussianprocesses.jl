module gaussianprocesses
export AbstractGaussianProcess, lnGP, nnGP

abstract type AbstractGaussianProcess end



"""
    struct lnGP <: AbstractGaussianProcess

Creates a Gaussian process with linear covariance function.

#Arguments
- `lthbounds` : a dictionary of pairs of the bounds on the hyperparameters in log10 space, such as Dict(0=> [0,6], 1=> [-3,4], 2=> [-6,-4])
- `x` : a 1-d array of the abscissa data
- `y` : a multi-dimensional array of the ordinate data
- `merrors` : if specified, a 1-d array of the measurement errors (as variances) v, else `nothing`
- `noparams` : Number of parameters
"""
mutable struct lnGP <: AbstractGaussianProcess
    b
    x
    y
    xnew
    merror
    noparams::Int
    description::String
    lnGP(lthbounds::Dict, x, y) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, nothing, 2, "linear gaussian process")
    lnGP(lthbounds::Dict, x, y, merrors) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, merror, 2, "linear gaussian process")
end


"""
    struct nnGP <: AbstractGaussianProcess

Creates a Gaussian process with neural network covariance function.

#Arguments
- `lthbounds` : a dictionary of pairs of the bounds on the hyperparameters in log10 space, such as Dict(0=> [0,6], 1=> [-3,4], 2=> [-6,-4])
- `x` : a 1-d array of the abscissa data
- `y` : a multi-dimensional array of the ordinate data
- `merrors` : if specified, a 1-d array of the measurement errors (as variances) v, else `nothing`
- `noparams` : Number of parameters
"""
mutable struct nnGP <: AbstractGaussianProcess
    b
    x
    y
    xnew
    merror
    noparams::Int
    description::String
    nnGP(lthbounds::Dict, x, y) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, nothing, 2, "neural network Gaussian process")
    nnGP(lthbounds::Dict, x, y, merrors) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, merror, 2, "neural network Gaussian process")
end


function info(self::nnGP)
    println("hparam[1] determines the initial value.")
    println("hparam[2] determines the flexibility.")
    println("hparam[3] determines the variance of the measurement error.")
end


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
mutable struct sqexpGP <: AbstractGaussianProcess
    b
    x
    y
    xnew
    merror
    noparams::Int
    description::String
    sqexpGP(lthbounds::Dict, x, y) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, nothing, 2, "squared exponential Gaussian process")
    sqexpGP(lthbounds::Dict, x, y, merrors) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, merror, 2, "squared exponential Gaussian process")
end


function info(self::sqexpGP)
    println("hparam[1] determines the amplitude of variation.")
    println("hparam[2] determines the flexibility.")
    println("hparam[3] determines the variance of the measurement error.")
end


"""
    struct sqexplinGP <: AbstractGaussianProcess

Creates a Gaussian process with squared exponential covariance function with a linear trend.

#Arguments
- `lthbounds` : a dictionary of pairs of the bounds on the hyperparameters in log10 space, such as Dict(0=> [0,6], 1=> [-3,4], 2=> [-6,-4])
- `x` : a 1-d array of the abscissa data
- `y` : a multi-dimensional array of the ordinate data
- `merrors` : if specified, a 1-d array of the measurement errors (as variances) v, else `nothing`
- `noparams` : Number of parameters
"""
mutable struct sqexplinGP <: AbstractGaussianProcess
    b
    x
    y
    xnew
    merror
    noparams::Int
    description::String
    sqexplinGP(lthbounds::Dict, x, y) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, nothing, 3, "squared exponential Gaussian process with a linear trend")
    sqexplinGP(lthbounds::Dict, x, y, merrors) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, merror, 3, "squared exponential Gaussian process with a linear trend")
end


function info(self::sqexplinGP)
    println("hparam[1] determines the amplitude of variation.")
    println("hparam[2] determines the flexibility.")
    println("hparam[3] determines the linear trend with increasing input.")
    println("hparam[4] determines the variance of the measurement error.")
end


"""
    struct maternGP <: AbstractGaussianProcess

Creates a Gaussian process with Matern covariance function that is twice differentiable.

#Arguments
- `lthbounds` : a dictionary of pairs of the bounds on the hyperparameters in log10 space, such as Dict(0=> [0,6], 1=> [-3,4], 2=> [-6,-4])
- `x` : a 1-d array of the abscissa data
- `y` : a multi-dimensional array of the ordinate data
- `merrors` : if specified, a 1-d array of the measurement errors (as variances) v, else `nothing`
- `noparams` : Number of parameters
"""
mutable struct maternGP <:AbstractGaussianProcess
    b
    x
    y
    xnew
    merror
    noparams::Int
    description::String
    maternGP(lthbounds::Dict, x, y) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, nothing, 2, "(twice differentiable) Matern covariance function")
    maternGP(lthbounds::Dict, x, y, merrors) = new([lthbounds[a] for a in keys(lthbounds)], x, y, x, merror, 2, "(twice differentiable) Matern covariance function")
end


function info(self::maternGP)
    println("hparam[1] determines the amplitude of variation.")
    println("hparam[2] determines the stiffness.")
    println("hparam[3] determines the variance of the measurement error.")
end
