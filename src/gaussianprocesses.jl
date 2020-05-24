module gaussianprocesses

using LinearAlgebra, Optim

export AbstractGaussianProcess, sqexpGP, covfn

abstract type AbstractGaussianProcess end



"""
    struct sqexpGP <: AbstractGaussianProcess

Creates a Gaussian process with squared exponential covariance function.

#Arguments
- `lthbounds` : 2 dimensional array Nx2 of bounds on the hyperparameters in log10 space, first column lower bounds, second column upper bounds, N parameters
- `x` : a 1-d array of the abscissa data
- `y` : a multi-dimensional array of the ordinate data
- `merrors` : if specified, a 1-d array of the measurement errors (as variances) v, else `nothing`
- `noparams` : Number of parameters
"""
mutable struct sqexpGP{
    T_b<:AbstractArray,
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

sqexpGP(lthbounds, x, y) = sqexpGP(lthbounds, x, y, x, Float64[], 2, "squared exponential Gaussian process")
sqexpGP(lthbounds, x, y, merrors) = sqexpGP(lthbounds, x, y, x, merrors, 2, "squared exponential Gaussian process")


function info(gp::sqexpGP)
    println("hparam[1] determines the amplitude of variation.")
    println("hparam[2] determines the flexibility.")
    println("hparam[3] determines the variance of the measurement error.")
end


"""
    covfn(gp::sqexpGP, x, xp, lth)

Squared exponential covariance function.
Returns the kernel function and Jacobian of the kernel function.

#Arguments
- `gp` : gaussianprocess with squared exponential covariance function
- `x` : a 1-d array of abscissa
- `xp` : a 1-d array of alternative abscissa
- `lth` : the log of the hyperparameters
"""
function covfn(gp::sqexpGP, x, xp, lth)
    th = exp.(lth)
    e = exp.(-th[2] / 2.0 * (x .- xp).^2)
    k = th[1] * e
    jk = zeros(length(xp), gp.noparams)
    jk[:, 1] = e * th[1]
    jk[:, 2] = -th[1] * th[2] * e / 2.0 .* (x .- xp).^2
    return k, jk
end


"""
    function kernelmatrix(gp::sqexpGP, lth, x)

Returns kernel matrix K(X,X) supplemented with measurement noise and its Cholesky decomposition.

#Arguments
- `gp` : AbstractGaussianProcess
- `lth` : log of the hyperparameters
- `x` : abscissa values
"""
function kernelmatrix(gp::AbstractGaussianProcess, lth, x)
    k = zeros(length(x), length(x))
    for i in 1:length(x)
        k[i,:]= covfn(gp, x[i], x, lth)[1]
    end

    if ~isempty(gp.merrors)
        kn = k .+ exp(lth[end]) * Diagonal(gp.merrors)
    else
        kn = k .+ exp(lth[end]) * I(length(x))
    end
    return k, kn
end


"""
    function nlml(gp::AbstractGaussianProcess, lth)

Returns negative of log marginal likelihood.

#Arguments
- `gp` : AbstractGaussianProcess
- `lth` : log of the hyperparameters
"""
function nlml(gp::AbstractGaussianProcess, lth)
    x, y = gp.x, gp.y
    k, kn = kernelmatrix(gp, lth, x)
    al = kn\y
    L = cholesky(kn).U
    halfdetK = Diagonal([L[i, i] for i in 1:size(L, 1)]) |> log |> sum
    return 0.5* dot(y, al) + halfdetK + 0.5 * length(y) * log(2 * pi)
end


"""
    function jacnlml(gp::AbstractGaussianProcess, lth)

Returns the Jacobian of negative log marginal likelihood with respect to the hyperparameters with deriviatives being taken assuming the hyperparmaters are in log space.

#Arguments
- `gp` : Gaussian Process
- `lth` : log of the hyperparameters
"""
function jacnlml(gp::AbstractGaussianProcess, lth)

    x, y = gp.x, gp.y
    k, L = kernelmatrix(gp, lth, x)
    # find derivatives of kernel matrix wrt hyperparameters
    kjac = zeros(length(x), length(x), length(lth))
    for i in 1:length(x)
        kjac[i, :, 1:end-1]= covfn(gp, x[i], x, lth)[2]
    end
    if ~isempty(gp.merrors)
        kjac[:, :, end]= Diagonal(gp.merrors) * exp(lth[end])
    else
        kjac[:, :, end]= I(length(x)) * exp(lth[end])
    end
    # calculate jacobian
    al = L\y
    alal = al .* al'
    Kinv = L\I(length(x))
    return [-0.5 * tr(dot(alal - Kinv, kjac[:, :, i])) for i in 1:length(lth)]
end


"""
    function jacnlml_optim(gp::AbstractGaussianProcess, lth)

Returns the Jacobian of negative log marginal likelihood with respect to the hyperparameters with deriviatives being taken assuming the hyperparmaters are in log space.
Written to be used for Optim.optimize.

#Arguments
- `gp` : Gaussian Process
- `lth` : log of the hyperparameters
"""
function jacnlml_optim!(gp::AbstractGaussianProcess, storage, lth)
    x, y = gp.x, gp.y
    k, L = kernelmatrix(gp, lth, x)
    # find derivatives of kernel matrix wrt hyperparameters
    kjac = zeros(length(x), length(x), length(lth))
    for i in 1:length(x)
        kjac[i, :, 1:end-1] = covfn(gp, x[i], x, lth)[2]
    end
    if ~isempty(gp.merrors)
        kjac[:, :, end] = Diagonal(gp.merrors) * exp(lth[end])
    else
        kjac[:, :, end] = I(length(x)) * exp(lth[end])
    end
    # calculate jacobian
    al = L\y
    alal = al .* al'
    Kinv = L\I(length(x))
    storage = [-0.5 * tr(dot(alal - Kinv, kjac[:, :, i])) for i in 1:length(lth)]
end


"""
    function findhyperparameters(
        gp::AbstractGaussianProcess;
        noruns=1,
        exitearly=false,
        stvals=[],
        optmethod='l_bfgs_b',
        optmessages=false,
        quiet=true,
        linalgmax=3)

Finds the best fit hyperparameters (.lth_opt) and the optimum value of negative log marginal likelihood (.nlml_opt).

#Arguments
- `noruns` : number of attempts to find optimum hyperparameters (the best of all the runs is chosen)
- `exitearly` : if True, fitting stops at the first successful attempt
- `stvals` : an (optional) initial guess for the log hyperparameters
- `optmethod` : the optimization routine to be used, either 'l_bfgs_b' (default) or 'tnc'
- `optmessages`: if True, display messages from the optimization routine
- `quiet`: if True, print warning that if an optimum hyperparameter is at a bound
- `linalgmax` : number of attempts (default is 3) if a linear algebra (numerical) error is generated
"""
function findhyperparameters(
    gp::AbstractGaussianProcess;
    noruns=1,
    exitearly=false,
    stvals=[],
    optmessages=false,
    quiet=true,
    linalgmax=3)

    if ~isempty(stvals)
        if length(stvals) != length(gp.b)
            throw(ArgumentError("If initial guess is given, give guess for all parameters"))
        end
    end

    b = gp.b
    lmlml = zeros(Float64, noruns)
    lthf = zeros(Float64, noruns, size(b, 1))
    success = zeros(Bool, noruns)

    # convert b into exponential base
    b .*= log(10)

    # run optimization
    for i in 1:noruns
        try
            if ~isempty(stvals)
                # initial values given for hyperparameters
                lth = stvals
            else
                # choose random initial values for hyperparameters
                lth = [b[j, 1] + (b[j, 2] - b[j, 1]) * rand() for j in 1:size(b, 1)]
            end

            # run Gaussian process
            f(x) = nlml(gp, x)
            g!(storage, x) = jacnlml_optim!(gp, storage, x)

            inner_optimizer = GradientDescent()

            res = optimize(f, lth)

            lthf[i,:] =  Optim.minimizer(res)
            success[i] = Optim.converged(res)
            lmlml[i]= Optim.minimum(res)
            if success[i] != 1
                @warn "Warning: optimization failed at run "*"$i"
            else
                if exitearly
                    break
                end
            end
        catch e
        end
    end

    ind = findall(x->x==1, success)
    if isempty(ind)
        throw(ErrorException("No run converged."))
    else
        lmlml= lmlml[ind]
        lthf= lthf[ind, :]

        lthb= lthf[argmin(lmlml), :]
        nlml_opt = minimum(lmlml)
    end

    return lthb, nlml_opt
end


end  # module
Main.gaussianprocesses
