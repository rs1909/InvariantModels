using ManifoldsBase
using Manifolds

import Base.zero
import Base.randn

using DynamicPolynomials
using MultivariatePolynomials

using TaylorSeries
using Tullio
using LinearAlgebra
#for kmeans
using Clustering

#------------------------------------------------------------------------------------------------------------------------------------------
#
# UTILITY FUNCTIONS 
#
#------------------------------------------------------------------------------------------------------------------------------------------

# creates a uniform grid in the [0,2pi) interval (of 2*n + 1 points)
# The 2*n+1 number is because we are using n Fourier frequencies (with sin & cos) plus a constant term
@doc raw"""
    getgrid(n::Integer)

Returns a uniform grid on the interval ``[0, 2 \pi)`` with ``2n + 1`` number of points. The end point ``2\pi`` is not part of the grid.

The number ``n`` corresponds to the number of Fourier harmonics that can be represented on the grid.
"""
function getgrid(n::Integer)
    return range(0.0, 4 * n * pi / (2 * n + 1), length=2 * n + 1)
end

# the interpolation function
function psi(theta, np)
    return iszero(theta) ? one(theta) : sin((np / 2) * theta) / (np * sin(theta / 2))
end

# creates a vector that interpolates ths scalar valued 'theta' to the uniform grid
# the grid must be created by getgrid(n)
function fourierInterplate(grid, theta::Number)
    psi2 = (x) -> psi(x, length(grid))
    return psi2.(theta .- grid)
end

function fourierInterplate(grid, theta::AbstractArray{T,1}) where {T}
    W = zeros(T, length(grid), length(theta))
    psi2 = (x) -> psi(x, length(grid))
    for k = 1:length(theta)
        W[:, k] .= psi2.(theta[k] .- grid)
    end
    return W
end

function FourierMatrix(grid)
    fourier_order = div(length(grid) - 1, 2)
    @assert length(grid) == 2 * fourier_order + 1 "Incorrect grid size"
    tr = [exp(-k * 1im * grid[l]) for k = -fourier_order:fourier_order, l = 1:length(grid)]
    return tr
end

# creates a matrix that shifts a periodic function sampled on 'grid'
# by 'omega' to the left and maps it back to 'grid'
# (S_o u)(q) = u(q-o)
function shiftOperator(grid, omega)
    fourier_order = div(length(grid) - 1, 2)
    SH = zeros(typeof(omega), length(grid), length(grid))
    for j = 1:length(grid), k = 1:length(grid)
        SH[j, k] = psi(grid[j] - grid[k] - omega, length(grid))
    end
    return SH
end

function shiftOperator(grid, theta, omega)
    return fourierInterplate(grid .- omega, theta)
end

# nd : dimensions
# grid: uniform grid onm [0,2*pi)
# omega: shift amount
function shiftMinusDiagonalOperator(grid, B, omega)
    SH = shiftOperator(grid, omega)
    TR = zeros(size(B, 1) * size(B, 3), size(B, 2) * size(B, 3))
    nd = size(B, 1)
    for j = 1:size(SH, 1), k = 1:size(SH, 2)
        TR[1+(j-1)*nd:j*nd, 1+(k-1)*nd:k*nd] .= SH[j, k] * Diagonal(I, nd) - I[j, k] * B[:, :, k]
    end
    return TR
end

function bigShiftOperator(nd, grid, omega)
    SH = shiftOperator(grid, omega)
    TR = zeros(nd * size(SH, 1), nd * size(SH, 2))
    for j = 1:size(SH, 1), k = 1:size(SH, 2)
        TR[1+(j-1)*nd:j*nd, 1+(k-1)*nd:k*nd] .= SH[j, k] * Diagonal(I, nd)
    end
    return TR
end

function transferOperator(grid, B, omega)
    SH = shiftOperator(grid, omega)
    TR = zeros(size(B, 1) * size(B, 3), size(B, 2) * size(B, 3))
    nd = size(B, 1)
    for j = 1:size(SH, 1), k = 1:size(SH, 2)
        TR[1+(j-1)*nd:j*nd, 1+(k-1)*nd:k*nd] .= SH[j, k] * B[:, :, k]
    end
    return TR
end

function adjointTransferOperator(grid, B, omega)
    SH = shiftOperator(grid, -omega)
    TR = zeros(size(B, 1) * size(B, 3), size(B, 2) * size(B, 3))
    nd = size(B, 1)
    for j = 1:size(SH, 1), k = 1:size(SH, 2)
        TR[1+(j-1)*nd:j*nd, 1+(k-1)*nd:k*nd] .= transpose(B[:, :, j]) * SH[j, k]
    end
    return TR
end

# derivatives

function diffpsi(k, np)
    return iszero(k) ? zero(sin(k)) : (-1)^k / sin(k * pi / np) / 2
end

function differentialOperator(grid, omega)
    fourier_order = div(length(grid) - 1, 2)
    SH = zeros(typeof(omega), length(grid), length(grid))
    for j = 1:length(grid), k = 1:length(grid)
        SH[j, k] = diffpsi(j - k, length(grid)) * omega
    end
    return SH
end

function bigDifferentialOperator(nd, grid, omega)
    SH = differentialOperator(grid, omega)
    TR = zeros(nd * size(SH, 1), nd * size(SH, 2))
    for j = 1:size(SH, 1), k = 1:size(SH, 2)
        TR[1+(j-1)*nd:j*nd, 1+(k-1)*nd:k*nd] .= SH[j, k] * Diagonal(I, nd)
    end
    return TR
end

function differentialMinusDiagonalOperator(grid, B, omega)
    SH = differentialOperator(grid, omega)
    TR = zeros(size(B, 1) * size(B, 3), size(B, 2) * size(B, 3))
    nd = size(B, 1)
    for j = 1:size(SH, 1), k = 1:size(SH, 2)
        TR[1+(j-1)*nd:j*nd, 1+(k-1)*nd:k*nd] .= I[j, k] * B[:, :, k] - SH[j, k] * Diagonal(I, nd)
    end
    return TR
end

#------------------------------------------------------------------------------------------------------------------------------------------
#
# [POINTWISE REPRESENTATION] quasi periodic POLYNOMIAL 
#
#------------------------------------------------------------------------------------------------------------------------------------------

# ndim: input dimensionality
# n:    output dimensionality
struct QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,𝔽} <: AbstractManifold{𝔽}
    mexp
    to_exp
    admissible
    D1row
    D1col
    D1val
    D2row
    D2col
    D2val
    M::AbstractManifold{𝔽}
    R::AbstractRetractionMethod
    VT::AbstractVectorTransportMethod
end

function PolyOrder(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    return max_polyorder
end

function DensePolyExponents(ndim::Integer, order::Integer; min_order=0)
    if min_order <= order
        @polyvar x[1:ndim]
        mx0 = monomials(x, min_order:order)
        return hcat([exponents(m) for m in mx0]...)
    else
        return zeros(typeof(order), ndim, 0)
    end
    nothing
end

@doc raw"""
    M = QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field::AbstractNumbers=ℝ; perp_idx=1:dim_in)
    
Creates a vector valued polynomial that also a periodic function of another variable.
```math
    \boldsymbol{y} = P(\boldsymbol{x}, \theta)
```
The parameters are
* `dim_out` number of output dimensions, i.e., the dimensionality of ``\boldsymbol{y}``
* `dim_in` number independent variables, i.e., the dimensionality of ``\boldsymbol{x}``
* `fourier_order` number of Fourier harmonics for the independent variable ``\theta``
* `min_polyorder` the lowest monomial order contained in polynomial ``P``
* `max_polyorder` the highest monomial order contained in polynomial ``P``
* `field` whether it is real valued `field=ℝ` or complex valued `field=ℂ`
* `perp_idx` the indices of idependent variables in ``\boldsymbol{x}`` that must appear at least linearly in each monomial. 
  For example if we have 4 variables and `perp_idx = [1 2]`, the monomial ``x_3 x_4`` is not part of the polynomial, because they do not contains any of ``x_1`` or ``x_2``. However, ``x_1 x_3`` is part of the polynomial, because ``1`` is among the indices given by `perp_idx`.
"""
function QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field::AbstractNumbers=ℝ; perp_idx=1:dim_in)
    # for the derivatives up to second order
    to_exp = DensePolyExponents(dim_in, max_polyorder; min_order=max(0, min_polyorder - 2))
    if length(unique(perp_idx)) == dim_in
        admissible = findall((dropdims(sum(to_exp, dims=1), dims=1) .>= min_polyorder))
    else
        println("QPPolynomial: restricting monomials.")
        admissible = findall((dropdims(sum(to_exp, dims=1), dims=1) .>= min_polyorder) .&&
                             (dropdims(sum(to_exp[perp_idx, :], dims=1), dims=1) .!= 0))
    end
    mexp = to_exp[:, admissible]
    D1row, D1col, D1val = FirstDeriMatrices(to_exp, mexp)
    D2row, D2col, D2val = SecondDeriMatrices(to_exp, mexp)
    M = Euclidean(dim_out, size(mexp, 2), 2 * fourier_order + 1, field=field)
    R = ExponentialRetraction()
    VT = ParallelTransport()
    return QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}(mexp, to_exp, admissible, D1row, D1col, D1val, D2row, D2col, D2val, M, R, VT)
end

function DenseMonomials(mexp, data)
    zp = reshape(data, size(data, 1), 1, :) .^ reshape(mexp, size(mexp, 1), :, 1)
    return dropdims(prod(zp, dims=1), dims=1)
end

@doc raw"""
    fromFunction!(M::QPPolynomial, W, fun)

Taylor expands Julia function `fun` to a polynomial [`QPPolynomial`](@ref), whose structure is prescribed by `M`. Function ``f`` = `fun` must have two arguments
```math
    f : \mathbb{R}^n \times [0,2\pi) \to \mathbb{R}^m,
```
where ``n`` is `dim_in` and ``m`` is the `dim_out` parameter of polynomial `M`. The result is copied into `W`.
"""
function fromFunction!(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, W, fun) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    x = set_variables("x", numvars=dim_in, order=max_polyorder)
    grid = getgrid(fourier_order)
    for j = 1:length(grid)
        y = fun(x, grid[j])
        for k = 1:dim_out
            for i = 1:size(M.mexp, 2)
                W[k, i, j] = getcoeff(y[k], M.mexp[:, i])
            end
        end
    end
    nothing
end

@doc raw"""
    W = fromFunction(M::QPPolynomial, fun)
    
Same as [`fromFunction!`](@ref), except that the result is returned in `W`.
"""
function fromFunction(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, fun) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    W = zero(M)
    fromFunction!(M, W, fun)
    return W
end

function fromData(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, thetaIN::Array{T,2}, dataIN, thetaOUT::Array{T,2}, dataOUT) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,T}
    MON = DenseMonomials(M.mexp, dataIN)
    #     @show size(thetaIN), size(dataIN), size(MON), size(dataOUT)
    @tullio X[j, k, l] := MON[j, l] * thetaIN[k, l]
    #     @show size(X)
    Xr = reshape(X, :, size(X, 3))
    YXt = dataOUT * transpose(Xr)
    XXt = Xr * transpose(Xr)
    Wr = YXt / XXt
    return reshape(Wr, size(Wr, 1), size(MON, 1), :)
end

@doc raw"""
    mapFromODE(M::QPPolynomial, W, fun, par, omega, dt, step)
    
Creates a Poincare map from the vector field given by `fun` and places it into the polynomial `M`, `X`. The parameters are
* `M` ,`W` the polynomial to contain the result.
* `fun` the vector field in the form of `fun(dx, x, p, theta)`, where `x` is the input state var, `dx` is the time-derivative ``\frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}t}``, `p` is the parameter, which equals to `par` and `theta` is the phase of forcing on the interval ``[0, 2\pi)``.
* `par` is the parameter (vector) of `fun`.
* `omega` is the forcing frequency. The independent variable `t` of `fun` is calculated as `theta` = ``\omega t``, where ``t`` is the independent variable (time).
* `dt` is the time step of the integrator.
* `step` is time period that the differential equation is solved for.
"""
function mapFromODE(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, W, fun, par, omega, dt, step) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    u0 = set_variables("x", numvars=dim_in, order=max_polyorder)
    grid = getgrid(fourier_order)
    for j = 1:length(grid)
        prob = ODEProblem((x, y, p, t) -> fun(x, y, p, omega * t), u0, (grid[j] / omega, grid[j] / omega + step), par)
        sol = solve(prob, ABM54(), dt=dt, internalnorm=(u, t) -> 0.0, adaptive=false)
        y = sol[end]
        for k = 1:dim_out
            for i = 1:size(M.mexp, 2)
                W[k, i, j] = getcoeff(y[k], M.mexp[:, i])
            end
        end
    end
    nothing
end

function thetaDerivative(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, X, omega) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    grid = getgrid(fourier_order)
    SH = differentialOperator(grid, omega)
    @tullio DX[i, j, p] := SH[p, q] * X[i, j, q]
    return DX
end

function zero(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    return zeros(dim_out, size(M.mexp, 2), 2 * fourier_order + 1)
end

function zero(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℂ}) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    return zeros(ComplexF64, dim_out, size(M.mexp, 2), 2 * fourier_order + 1)
end

function randn(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, p=nothing) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    return randn(dim_out, size(M.mexp, 2), 2 * fourier_order + 1)
end

function randn(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℂ}, p=nothing) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    return randn(ComplexF64, dim_out, size(M.mexp, 2), 2 * fourier_order + 1)
end


function InterpolationWeights(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, theta) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    #     out = zeros(ComplexF64, 2*fourier_order+1, length(theta))
    grid = getgrid(fourier_order)
    return fourierInterplate(grid, theta)
end

function makeCache(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, X, theta::Matrix, data::Matrix) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    to_MON = DenseMonomials(M.to_exp, data)
    MON = to_MON[M.admissible, :]
    #     @show size(X), size(MON), size(theta)
    @tullio res[i, k] := X[i, p, q] * MON[p, k] * theta[q, k]
    #     @show size(res) 
    return (MON=MON, to_MON=to_MON, val=res)
end

function updateCache!(cache, M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, X, theta::Matrix, data::Matrix, upmon=true) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    if upmon
        to_MON = DenseMonomials(M.to_exp, data)
        MON = to_MON[M.admissible, :]
        cache.MON .= MON
        cache.to_MON .= to_MON
    end
    MON = cache.MON
    @tullio res[i, k] := X[i, p, q] * MON[p, k] * theta[q, k]
    cache.val .= res
    return nothing
end

@doc raw"""
    Eval!(res, M::QPPolynomial, X, theta::Matrix, data::Matrix; cache=makeCache(M, X, theta, data))
    
Evaluate the polynomial `M`, `X` at multiple data points. The data points are given by `theta` and `data`. Each column of `theta` and `data` corresponds to a single data point. The result is copied into `res`.

The matrix `theta` contains the interpolation weights of Fourier collocation and not the actual values of ``\theta``. Generally, these interpolation weights are pre-calculated, which avoids repeated calculation. The number of rows of ``\theta`` is the same as ``2 n + 1``, where ``n`` is the number of Fourier modes resolved by polynomial `M`.
"""
function Eval!(res, M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, X, theta::Matrix, data::Matrix;
    cache=makeCache(M, X, theta, data)) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    res .= cache.val
    return res
end

@doc raw"""
    res = Eval(M::QPPolynomial, X, theta::Matrix, data::Matrix; cache=makeCache(M, X, theta, data))
    
Same as [`Eval!`](@ref), except that the result is returned in `res`. 
"""
function Eval(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, X, theta::Matrix, data::Matrix;
    cache=makeCache(M, X, theta, data)) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    return cache.val
end

function Eval(M::QPPolynomial, X, theta::Array{T,1}, data::Array{T,2}) where {T}
    return Eval(M, X, InterpolationWeights(M, theta), data)
end

# vector valued
function Eval(M::QPPolynomial, X, theta::Number, data::Vector)
    return vec(Eval(M, X, InterpolationWeights(M, [theta]), reshape(data, :, 1)))
end

function L0_DF!(DX, M::QPPolynomial, X, theta, data, L0; cache=makeCache(M, X, theta, data))
    MON = cache.MON
    #     @tullio DX[i,p,q] = L0[i,k] * MON[p,k] * theta[q,k]
    @inbounds @fastmath for i in axes(L0, 1), p in axes(MON, 1), q in axes(theta, 1), k in axes(theta, 2)
        DX[i, p, q] += L0[i, k] * MON[p, k] * theta[q, k]
    end
    return DX
end

# gradient with respect to the parameter
function L0_DF(M::QPPolynomial, X, theta, data, L0; cache=makeCache(M, X, theta, data))
    MON = cache.MON
    #     DX = zeros(eltype(theta), size(L0,1), size(MON,1), size(theta,1))
    #     @inbounds @fastmath for i in axes(L0,1), p in axes(MON,1), q in axes(theta,1), k in axes(theta,2)
    #         DX[i,p,q] += L0[i,k] * MON[p,k] * theta[q,k]
    #     end
    #     @show size(L0), size(MON), size(theta)
    @tullio DX[i, p, q] := L0[i, k] * MON[p, k] * theta[q, k]
    return DX
end

function DF_dt(M::QPPolynomial, X, theta, data; dt, cache=makeCache(M, X, theta, data))
    MON = cache.MON
    @tullio res[i, k] := dt[i, p, q] * MON[p, k] * theta[q, k]
    #     res = zeros(eltype(dt), size(dt,1), size(MON,2))
    #     @inbounds @fastmath for i in axes(dt,1), p in axes(dt,2), q in axes(dt,3), k in axes(MON,2) 
    #         res[i,k] += dt[i,p,q] * MON[p,k] * theta[q,k]
    #     end
    return res
end

function DF_DF_scale(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, X,
    theta, data, scale; cache=makeCache(M, X, theta, data)) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    MON = cache.MON
    id = Diagonal(I, dim_in)
    @tullio HESS[i1, p1, q1, i2, p2, q2] := id[i1, i2] * MON[p2, k] * theta[q2, k] * MON[p1, k] * theta[q1, k] * scale[k]
    return HESS
end

function JF_helper!(res, X, to_MON, theta, rowA, colA, factA)
    #     @inbounds 
    @fastmath for r in axes(rowA, 2), p in axes(rowA, 1), q in axes(theta, 1), i in axes(X, 1)
        a = (X[i, rowA[p, r], q] * factA[p, r])
        for k in axes(theta, 2)
            res[i, r, k] += a * to_MON[colA[p, r], k] * theta[q, k]
        end
    end
    nothing
end

function JF(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, X, theta, data; cache=makeCache(M, X, theta, data)) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    to_MON = cache.to_MON
    res = zeros(eltype(X), dim_out, dim_in, size(data, 2))
    JF_helper!(res, X, to_MON, theta, M.D1row, M.D1col, M.D1val)
    return res
end

function L0_JF_helper!(res, L0, X, to_MON, theta, rowA, colA, factA)
    @inbounds @fastmath for r in axes(rowA, 2), p in axes(rowA, 1), q in axes(theta, 1), i in axes(X, 1)
        a = (X[i, rowA[p, r], q] * factA[p, r])
        for k in axes(theta, 2)
            res[r, k] += L0[i, k] * a * to_MON[colA[p, r], k] * theta[q, k]
        end
    end
    nothing
end

# Jacobian multiplied from the right
function L0_JF(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, X, theta, data, L0; cache=makeCache(M, X, theta, data)) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    to_MON = cache.to_MON
    res = zeros(eltype(X), dim_in, size(data, 2))
    #     print("L0_JF:")
    #     @time 
    L0_JF_helper!(res, L0, X, to_MON, theta, M.D1row, M.D1col, M.D1val)
    return res
end

function test_L0_JF(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, X, theta, data, L0; cache=nothing, eps=1e-6) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    res = zeros(eltype(X), dim_in, size(data, 2))
    dataE = copy(data)
    for k = 1:size(data, 1)
        dataE[k, :] .+= eps
        res[k, :] .= dropdims(sum(L0 .* (Eval(M, X, theta, dataE) - Eval(M, X, theta, data)), dims=1), dims=1) ./ eps
        dataE[k, :] .= data[k, :]
    end
    return res
end

function JF_dx_helper!(res, X, to_MON, theta, dx, rowA, colA, factA)
    @inbounds @fastmath for r = 1:size(rowA, 2), p = 1:size(rowA, 1)
        for q = 1:size(theta, 1), i = 1:size(X, 1)
            a = (X[i, rowA[p, r], q] * factA[p, r])
            for k = 1:size(theta, 2)
                res[i, k] += a * to_MON[colA[p, r], k] * theta[q, k] * dx[r, k]
            end
        end
    end
    nothing
end

# Jacobian multiplied from the left
function JF_dx(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, X, theta, data, dx; cache=makeCache(M, X, theta, data)) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    to_MON = cache.to_MON
    res = zeros(eltype(X), size(X, 1), size(data, 2))
    #     print("JF_dx:")
    #     @time 
    JF_dx_helper!(res, X, to_MON, theta, dx, M.D1row, M.D1col, M.D1val)
    return res
end

function L0_HF_dx_helper!(hess, L0, X, to_MON, theta, dx, rowA, colA, factA)
    @inbounds @fastmath for s = 1:size(rowA, 2), r = 1:size(rowA, 2), p = 1:size(rowA, 1)
        for q = 1:size(theta, 1), i = 1:size(X, 1),
            a = (X[i, rowA[p, r, s], q] * factA[p, r, s])

            for k = 1:size(theta, 2)
                hess[r, k] += L0[i, k] * a * to_MON[colA[p, r, s], k] * theta[q, k] * dx[s, k]
            end
        end
    end
    nothing
end

# L0 H {dx, .} -> is a vector
function L0_HF_dx(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, X, theta, data, L0, dx; cache=makeCache(M, X, theta, data)) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    to_MON = cache.to_MON
    hess = zeros(eltype(X), dim_in, size(data, 2))
    #     print("L0_HF:")
    #     @time 
    L0_HF_dx_helper!(hess, L0, X, to_MON, theta, dx, M.D2row, M.D2col, M.D2val)
    return hess
end

function L0_HF_helper!(hess, L0, X, to_MON, theta, rowA, colA, factA)
    @inbounds @fastmath for s = 1:size(rowA, 2), r = 1:size(rowA, 2), p = 1:size(rowA, 1)
        for q in axes(theta, 1), i in axes(X, 1),
            a = (X[i, rowA[p, r, s], q] * factA[p, r, s])

            for k in axes(theta, 2)
                hess[r, s, k] += L0[i, k] * a * to_MON[colA[p, r, s], k] * theta[q, k]
            end
        end
    end
    nothing
end

# L0 H {dx, .} -> is a vector
function L0_HF(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, X, theta, data, L0; cache=makeCache(M, X, theta, data)) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    to_MON = cache.to_MON
    hess = zeros(eltype(X), dim_in, dim_in, size(data, 2))
    #     print("L0_HF:")
    #     @time 
    L0_HF_helper!(hess, L0, X, to_MON, theta, M.D2row, M.D2col, M.D2val)
    return hess
end

# returns the first index of mexp, which equals to iexp
# defined in polymethods.jl
function PolyFindIndex(mexp, iexp)
    return findfirst(dropdims(prod(mexp .== iexp, dims=1), dims=1))
end

# where are the indices of from_exp in to_exp?
function PolyIndexMap(to_mexp, from_exp)
    index_map = zeros(eltype(from_exp), size(from_exp, 2))
    for k = 1:size(from_exp, 2)
        index_map[k] = PolyFindIndex(to_mexp, from_exp[:, k])
    end
    return index_map
end

# creates matrices indexed by the variable differentiated against
#   row[k,r], col[k,r], val[k,r].
# where r is the variable differentiated against
# normally
#   sum_k X[i,k] * imon[k,:]
# goes to
#   X[i,row[k,r]] * val[k,r] * omon[col[k,r],:]
# if we are to create coordinates Y
#   Y[:,col[k,r],:] .= val[k,r] * X[:,row[k,r],:] ( for k in axes(val,1) ; r is D_r )
function FirstDeriMatrices(to_exp, from_exp)
    rowA = Array{Array{Int,1},1}(undef, size(from_exp, 1))
    colA = Array{Array{Int,1},1}(undef, size(from_exp, 1))
    valA = Array{Array{Int,1},1}(undef, size(from_exp, 1))
    for var = 1:size(from_exp, 1)
        row = Array{Int,1}(undef, 0)
        col = Array{Int,1}(undef, 0)
        val = Array{Int,1}(undef, 0)
        for k = 1:size(from_exp, 2)
            id = copy(from_exp[:, k]) # this is a copy not a view
            if id[var] > 0
                id[var] -= 1
                x = PolyFindIndex(to_exp, id)
                if x != nothing
                    push!(row, k)
                    push!(col, x)
                    push!(val, from_exp[var, k])
                end
            end
        end
        rowA[var] = row
        colA[var] = col
        valA[var] = val
    end
    mxl = maximum([length(vv) for vv in rowA])
    row = ones(Int, mxl, length(rowA))
    col = ones(Int, mxl, length(rowA))
    val = zeros(Int, mxl, length(rowA))
    for r = 1:size(row, 2)
#         @show size(row[:, r]), size(rowA[r])
        cl = length(rowA[r])
        row[1:cl, r] .= rowA[r]
        col[1:cl, r] .= colA[r]
        val[1:cl, r] .= valA[r]
    end
    return row, col, val
end

function SecondDeriMatrices(to_exp, from_exp)
    rowA = Array{Array{Int,1},2}(undef, size(from_exp, 1), size(from_exp, 1))
    colA = Array{Array{Int,1},2}(undef, size(from_exp, 1), size(from_exp, 1))
    valA = Array{Array{Int,1},2}(undef, size(from_exp, 1), size(from_exp, 1))
    for p = 1:size(from_exp, 1), q = 1:size(from_exp, 1)
        row = Array{Int,1}(undef, 0)
        col = Array{Int,1}(undef, 0)
        val = Array{Int,1}(undef, 0)
        for k = 1:size(from_exp, 2)
            id = copy(from_exp[:, k]) # this is a copy not a view
            if id[p] > 0
                id[p] -= 1
                if id[q] > 0
                    id[q] -= 1
                    x = PolyFindIndex(to_exp, id)
                    if x != nothing
                        push!(row, k)
                        push!(col, x)
                        if p == q
                            push!(val, from_exp[p, k] * (from_exp[p, k] - 1))
                        else
                            push!(val, from_exp[p, k] * from_exp[q, k])
                        end
                    end
                end
            end
        end
        rowA[p, q] = row
        colA[p, q] = col
        valA[p, q] = val
    end
    mxl = maximum([length(vv) for vv in rowA])
    row = ones(Int, mxl, size(from_exp, 1), size(from_exp, 1))
    col = ones(Int, mxl, size(from_exp, 1), size(from_exp, 1))
    val = zeros(Int, mxl, size(from_exp, 1), size(from_exp, 1))
    for p = 1:size(from_exp, 1), q = 1:size(from_exp, 1)
        cl = length(rowA[p,q])
        row[1:cl, p, q] .= rowA[p, q]
        col[1:cl, p, q] .= colA[p, q]
        val[1:cl, p, q] .= valA[p, q]
    end
    return row, col, val
end

function GetConstantPart(M::QPPolynomial, X)
    cid = findfirst(isequal(0), dropdims(sum(M.mexp, dims=1), dims=1))
    if cid != nothing
        return X[:, cid, :]
    end
    return zero(X[:, 1, :])
end

function SetConstantPart!(M::QPPolynomial, X, C)
    cid = findfirst(isequal(0), dropdims(sum(M.mexp, dims=1), dims=1))
    if cid != nothing
        X[:, cid, :] .= C
    end
    return nothing
end

function GetLinearPart(M::QPPolynomial, X)
    linear_indices = findall(dropdims(sum(M.mexp, dims=1), dims=1) .== 1)
    to_indices = [findfirst(M.mexp[:, k] .== 1) for k in linear_indices]
    B = zeros(eltype(X), size(X, 1), length(linear_indices), size(X, 3))
    for k = 1:size(X, 1), l = 1:size(M.mexp, 1)
        B[k, to_indices[l], :] .= X[k, linear_indices[l], :]
    end
    return B
end

function SetLinearPart!(M::QPPolynomial, X, B)
    linear_indices = findall(dropdims(sum(M.mexp, dims=1), dims=1) .== 1)
    to_indices = [findfirst(M.mexp[:, k] .== 1) for k in linear_indices]
    for k = 1:size(X, 1), l = 1:size(M.mexp, 1)
        X[k, linear_indices[l], :] .= B[k, to_indices[l], :]
    end
    return nothing
end

function SetLinearPart!(M::QPPolynomial, X, B::Number)
    linear_indices = findall(dropdims(sum(M.mexp, dims=1), dims=1) .== 1)
    to_indices = [findfirst(M.mexp[:, k] .== 1) for k in linear_indices]
    for k = 1:size(X, 1), l = 1:size(M.mexp, 1)
        X[k, linear_indices[l], :] .= B
    end
    return nothing
end

function copyMost!(MD::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, XD, MS::QPPolynomial, XS) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    for k in axes(MD.mexp, 2)
        idx = PolyFindIndex(MS.mexp, MD.mexp[:, k])
        if idx != nothing
            XD[:, k, :] .= XS[:, idx, :]
        end
    end
    nothing
end

function PointwisePolyMul!(out::AbstractArray{T,2}, in1::AbstractArray{T,2}, in2::AbstractArray{T,2}, multab) where {T}
    fourier_order1 = div(size(in1, 2) - 1, 2)
    fourier_order2 = div(size(in2, 2) - 1, 2)
    fourier_order_out = div(size(out, 2) - 1, 2)
    @assert size(in1, 2) == 2 * fourier_order1 + 1 "Incorrect grid size"
    @assert size(in2, 2) == 2 * fourier_order2 + 1 "Incorrect grid size"
    @assert size(out, 2) == 2 * fourier_order_out + 1 "Incorrect grid size"
    @assert fourier_order_out == fourier_order1 "Non-matching grid size"
    @assert fourier_order_out == fourier_order2 "Non-matching grid size"
    #     @show fourier_order_out, fourier_order1, fourier_order2

    for k = 1:size(multab, 1)
        for p = 1:size(in1, 2)
            out[multab[k, 3], p] += in1[multab[k, 1], p] * in2[multab[k, 2], p]
        end
    end
    return nothing
end

# function PolyMul!(Mout::QPPolynomial, Xout, Min1::QPPolynomial, Xin1, Min2::QPPolynomial, Xin2, multab = mulTable(Mout.mexp, Min1.mexp, Min2.mexp))
#     PointwisePolyMul!(Xout, Xin1, Xin2, multab)
# end

function QPPolynomialSubstitute!(Mout::QPPolynomial{dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder,field}, Xout,
    M1::QPPolynomial{dim_out,dim_out1,fourier_order1,min_polyorder1,max_polyorder1,field}, X1,
    M2::QPPolynomial{dim_out1,dim_in,fourier_order2,min_polyorder2,max_polyorder2,field}, X2, omega1=0, omega2=0) where
{dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder,field,
    dim_out1,fourier_order1,min_polyorder1,max_polyorder1,
    fourier_order2,min_polyorder2,max_polyorder2}
    # create an output that has constant terms
    MOC = QPPolynomial(1, dim_in, fourier_order_out, 0, max_polyorder, field)
    to = zero(MOC)
    res = zero(MOC)
    #     fourier_order1 = div(size(X1,3)-1,2)
    # where are the indices of Mout.mexp within MOC.mexp: XOC[:,index_map,:] .= Xout 
    index_map = PolyIndexMap(MOC.mexp, Mout.mexp)
    tab = mulTable(MOC.mexp, MOC.mexp, M2.mexp)
    interp1 = shiftOperator(getgrid(fourier_order1), getgrid(fourier_order_out), omega1) # corrected -omega
    interp2 = shiftOperator(getgrid(fourier_order2), getgrid(fourier_order_out), omega2) # corrected -omega

    @tullio X1interp[j, k, p] := interp1[q, p] * X1[j, k, q]
    @tullio X2interp[j, k, p] := interp2[q, p] * X2[j, k, q]
    # index of constant in the output
    Xout .= 0
    cfc = findfirst(dropdims(sum(MOC.mexp, dims=1), dims=1) .== 0)
    # substitute into all monomials
    for d = 1:size(X1, 1) # all dimensions
        for k = 1:size(M1.mexp, 2) # all monomials
            to .= zero(eltype(to))
            to[1, cfc, :] .= X1interp[d, k, :] # the constant coefficient
            # l select the variable in the monomial
            for l = 1:size(M1.mexp, 1)
                # multiply the exponent times
                for p = 1:M1.mexp[l, k]
                    # should not add to the previous result
                    res .= 0
                    @views PointwisePolyMul!(res[1, :, :], to[1, :, :], X2interp[l, :, :], tab)
                    to .= res
                end
            end
            Xout[d, :, :] .+= to[1, index_map, :]
        end
    end
    return nothing
end

# function QPPolynomialShift(M::QPPolynomial{dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field}, X, omega) where {dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field}
#     grid = range(0,2*pi,length=2*fourier_order+2)[1:end-1]
#     SH = shiftOperator(grid, omega)
#     @tullio XSH[p,q,k] := SH[k,j] * X[p,q,j]
#     return XSH
# end

# create a transfer operator out of the linear part of a polynomial
function transferOperator(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, X, omega) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    @assert dim_in == dim_out "The polynomial should be X -> X and not X -> Y"
    B = GetLinearPart(M, X)

    return transferOperator(range(0, 2 * pi, length=2 * fourier_order + 2)[1:end-1], B, omega)
end

function inverse(M::QPPolynomial{dim_out,dim_out,fourier_order,min_polyorder,max_polyorder,ℝ}, X) where {dim_out,fourier_order,min_polyorder,max_polyorder}
    A = GetLinearPart(M, X)
    Xn = deepcopy(X)
    Xn *= -1
    SetLinearPart!(M, Xn, zero(A))
    XId = zero(M)
    # need this dance, because we do not know which column of XId is which
    XIp = zero(M)
    for k in axes(XIp, 3)
        XIp[:, :, k] .= Array(I, size(XIp, 1), size(XIp, 2))
    end
    SetLinearPart!(M, XId, XIp)
    #
    Xres = zero(M)
    R = zero(M)
    Xsub = zero(M)
    for k = 1:max_polyorder+1
        QPPolynomialSubstitute!(M, Xsub, M, Xn, M, Xres)
        Xsub .+= XId
        for k in axes(Xres, 3)
            R[:, :, k] .= A[:, :, k] \ Xsub[:, :, k]
        end
        @show maximum(abs.(Xres - R))
        if maximum(abs.(Xres - R)) <= 8 * eps(eltype(R))
            break
        end
        Xres .= R
    end
    return Xres
end

# ONLY FOR FREQUENCY CALCULATION: in ScalingFunctions

# @doc raw"""
#     (P)^T P is a scalar
# """
# function DensePolySquared!(Mout::QPPolynomial, Xout, Min::QPPolynomial, Xin)
#     multab = mulTable(Mout.mexp, Min.mexp, Min.mexp)
#     Xout .= 0
#     # this is a matrix-vector multiplication
#     for p = 1:size(Xin,1) # number of rows in input
#         @views PointwisePolyMul!(Xout[1,:,:], Xin[p,:,:], Xin[p,:,:], multab)
#     end
#     return nothing
# end
# 
# @doc raw"""
#     (D_P)^T P is a vector
#     make sure that the minimum order of the input is zero!
# """
# function DensePolyDeriTransposeMul!(Mout::QPPolynomial, Xout, Min::QPPolynomial, Xin)
#     multab = mulTable(Mout.mexp, Min.mexp, Min.mexp)
#     row, col, val = FirstDeriMatrices(Min.mexp, Min.mexp)
#     Xout .= 0
#     # this is a matrix-vector multiplication
#     deri_r = zero(Xin[1,:,:])
#     for r = 1:size(Min.mexp,1)
#         for p = 1:size(Xin,1) # number of rows in input
#             deri_r .= zero(eltype(deri_r))
#             for k in axes(val,1)
#                 deri_r[col[k,r],:] .= val[k,r] * Xin[p,row[k,r],:]
#             end
#             @views PointwisePolyMul!(Xout[r,:,:], deri_r, Xin[p,:,:], multab)
#         end
#     end
#     return nothing
# end
# 
# 
# @doc raw"""
#     (D_P)^T D_P is a matrix
#     
#     make note that the output type as a manifold does not exist, so care needs to be taken.
#     It is assumed that the dimensionality is the same as the number of variables of Mout.
#     
#     make sure that the minimum order of the input is zero!
# """
# function DensePolyJabobianSquared!(Mout::QPPolynomial, Xout, Min::QPPolynomial, Xin)
#     multab = mulTable(Mout.mexp, Min.mexp, Min.mexp)
#     DM = DensePolyDeriMatrices(Min.mexpALL, Min.mexpALL)
#     Xout .= 0
#     # this is a matrix-matrix multiplication
#     for k = 1:size(Min.mexpALL,1)
#         for l = 1:size(Min.mexpALL,1) # number of variables in input, hence columns of derivative
#             for p = 1:size(Xin,1) # number of rows in input
#                 @views deri_k = DM[k]' * Xin[p,:]
#                 @views deri_l = DM[l]' * Xin[p,:]
#                 @views PointwisePolyMul!(Xout[k,l,:], vec(deri_k), vec(deri_l), multab)
#             end
#         end
#     end
#     return nothing
# end

#------------------------------------------------------------------------------------------------------------------------------------------
#
# [FOURIER REPRESENTATION] quasi periodic POLYNOMIAL
#
#------------------------------------------------------------------------------------------------------------------------------------------

# this is needed for normal form calculation OR analytic invariant foliation calculations

struct QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder} <: AbstractManifold{ℂ}
    mexp
    D1row
    D1col
    D1val
    M::AbstractManifold{ℂ}
    R::AbstractRetractionMethod
    VT::AbstractVectorTransportMethod
end

function PolyOrder(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    return max_polyorder
end

function PolyOrderIndices(M::Union{QPFourierPolynomial,QPPolynomial}, ord)
    return findall(isequal(ord), dropdims(sum(M.mexp, dims=1), dims=1))
end

function QPFourierPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder)
    mexp = DensePolyExponents(dim_in, max_polyorder; min_order=min_polyorder)
    M = Euclidean(dim_out, size(mexp, 2), 2 * fourier_order + 1, field=ℂ)
    R = ExponentialRetraction()
    VT = ParallelTransport()
    D1row, D1col, D1val = FirstDeriMatrices(mexp, mexp)
    return QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}(mexp, D1row, D1col, D1val, M, R, VT)
end

function zero(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    return zeros(ComplexF64, dim_out, size(M.mexp, 2), 2 * fourier_order + 1)
end

# creates a random one, which transforms back to a real signal
function randn(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, p=nothing) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    X = randn(dim_out, size(M.mexp, 2), 2 * fourier_order + 1)
    FM = FourierMatrix(range(0, 2 * pi, length=2 * fourier_order + 2)[1:end-1])
    @tullio Xout[i, j, k] := X[i, j, p] * FM[k, p] / (2 * fourier_order + 1)
    return Xout
end

@doc raw"""
    MF, XF = toFourier(M::QPPolynomial, X)
    
Converts the polynomial `M`, `X` to another polynomial that contains Fourier coefficients as opposed to collocated values of the polynomial.
"""
function toFourier(M::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, X) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    FM = FourierMatrix(range(0, 2 * pi, length=2 * fourier_order + 2)[1:end-1])
    @tullio Xout[i, j, k] := X[i, j, p] * FM[k, p] / (2 * fourier_order + 1)
    return QPFourierPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder), Xout
end

function fromFourierComplex(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, X) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    FM = FourierMatrix(range(0, 2 * pi, length=2 * fourier_order + 2)[1:end-1])'
    @tullio Xout[i, j, k] := X[i, j, p] * FM[k, p]
    return QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, ℂ), Xout
end

function fromFourier(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, X) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    Mo, Xo = fromFourierComplex(M, X)
    @show maximum(abs.(imag.(Xo)))
    #     @assert maximum(abs.(imag.(Xout))) < 8*eps(real(eltype(Xout))) "Not a REAL Fourier series. $(maximum(abs.(imag.(Xout))))"
    return QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder), real.(Xo)
end

function thetaDerivative(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, X, omega) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    DX = zero(X)
    for i in axes(X, 1), j in axes(X, 2), p in axes(X, 3)
        f = p - 1 - fourier_order
        DX[i, j, p] = 1im * f * X[i, j, p] * omega
    end
    return DX
end

function GetLinearPart(M::QPFourierPolynomial, X)
    linear_indices = findall(dropdims(sum(M.mexp, dims=1), dims=1) .== 1)
    to_indices = [findfirst(M.mexp[:, k] .== 1) for k in linear_indices]
    B = zeros(eltype(X), size(X, 1), length(linear_indices), size(X, 3))
    for k = 1:size(X, 1), l = 1:size(M.mexp, 1)
        B[k, to_indices[l], :] .= X[k, linear_indices[l], :]
    end
    B
end

function SetLinearPart!(M::QPFourierPolynomial, X, B)
    linear_indices = findall(dropdims(sum(M.mexp, dims=1), dims=1) .== 1)
    to_indices = [findfirst(M.mexp[:, k] .== 1) for k in linear_indices]
    for k = 1:size(X, 1), l = 1:size(M.mexp, 1)
        X[k, linear_indices[l], :] .= B[k, to_indices[l], :]
    end
    nothing
end

function GetConstantPart(M::QPFourierPolynomial, X)
    cid = findfirst(isequal(0), dropdims(sum(M.mexp, dims=1), dims=1))
    if cid != nothing
        return X[:, cid, :]
    end
    return zero(X[:, 1, :])
end

function SetConstantPart!(M::QPFourierPolynomial, X, C)
    cid = findfirst(isequal(0), dropdims(sum(M.mexp, dims=1), dims=1))
    if cid != nothing
        X[:, cid, :] .= C
    end
    return nothing
end

# polynomial substitution

@doc raw"""
    tab = mulTable(oexp, in1exp, in2exp)
    
Creates a list of triplets `[i1, i2, o]`, where `i1`, `i2` are the input monomial, which produce `o` output monomial.
"""
function mulTable(oexp, in1exp, in2exp)
    res = []
    od = maximum(sum(oexp, dims=1))
    p1 = sum(in1exp, dims=1)
    p2 = sum(in2exp, dims=1)
    pexp = zero(in1exp[:, 1])
    for k1 = 1:size(in1exp, 2)
        for k2 = 1:size(in2exp, 2)
            if p1[k1] + p2[k2] <= od
                pexp[:] = in1exp[:, k1] + in2exp[:, k2]
                idx = PolyFindIndex(oexp, pexp)
                push!(res, [k1, k2, idx])
            end
        end
    end
    out = zeros(typeof(od), length(res), 3)
    for k = 1:length(res)
        out[k, :] = res[k]
    end
    return out
end

@doc raw"""
    PolyMul!(out, in1, in2, multab)
    
Multiplies two scalar valued polynomials `in1`, `in2` and adds the result to `out`. 
All inputs are one-dimensional arrays, `multab` is produced by `mulTable`.
"""
function FourierPolyMul!(out::AbstractArray{T,2}, in1::AbstractArray{T,2}, in2::AbstractArray{T,2}, multab) where {T}
    fourier_order1 = div(size(in1, 2) - 1, 2)
    fourier_order2 = div(size(in2, 2) - 1, 2)
    fourier_order_out = div(size(out, 2) - 1, 2)
    @assert size(in1, 2) == 2 * fourier_order1 + 1 "Incorrect grid size"
    @assert size(in2, 2) == 2 * fourier_order2 + 1 "Incorrect grid size"
    @assert size(out, 2) == 2 * fourier_order_out + 1 "Incorrect grid size"

    for k = 1:size(multab, 1)
        for p = 1:size(in1, 2), q = 1:size(in2, 2)
            f1 = p - 1 - fourier_order1
            f2 = q - 1 - fourier_order2
            r = fourier_order_out + 1 + f1 + f2
            if (r > 0) && (r <= size(out, 2))
                out[multab[k, 3], r] += in1[multab[k, 1], p] * in2[multab[k, 2], q]
            end
        end
    end
    return nothing
end

function QPFourierSubstituteTabs(Mout::QPFourierPolynomial{dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder}, Xout,
    M1::QPFourierPolynomial, X1, M2::QPFourierPolynomial, X2) where {dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder}
    MOC = QPFourierPolynomial(1, dim_in, fourier_order_out, 0, max_polyorder)
    tab = mulTable(MOC.mexp, MOC.mexp, M2.mexp)
    index_map = PolyIndexMap(MOC.mexp, Mout.mexp)

    return MOC, tab, index_map
end

function QPFourierPolynomialSubstitute!(Mout::QPFourierPolynomial{dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder}, Xout,
    M1::QPFourierPolynomial, X1,
    M2::QPFourierPolynomial, X2, MOC, tab, index_map; shift=0.0) where {dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder}
    # create an output that has constant terms
    #     MOC = QPFourierPolynomial(1, dim_in, fourier_order_out, 0, max_polyorder)
    to = zero(MOC)
    res = zero(MOC)
    fourier_order1 = div(size(X1, 3) - 1, 2)
    # where are the indices of Mout.mexp within MOC.mexp: XOC[:,index_map,:] .= Xout 
    #     index_map = PolyIndexMap(MOC.mexp, Mout.mexp)
    #     tab = mulTable(MOC.mexp, MOC.mexp, M2.mexp)
    # index of constant in the output
    Xout .= 0
    cfc = findfirst(dropdims(sum(MOC.mexp, dims=1), dims=1) .== 0)
    # substitute into all monomials
    for d = 1:size(X1, 1) # all dimensions
        for k = 1:size(M1.mexp, 2) # all monomials
            to .= 0
            if fourier_order_out > fourier_order1
                to[1, cfc, :] .= zero(eltype(to))
                to[1, cfc, fourier_order_out+1-fourier_order1:fourier_order_out+1+fourier_order1] .= X1[d, k, :] # the constant coefficient
            else
                to[1, cfc, :] .= X1[d, k, fourier_order1+1-fourier_order_out:fourier_order1+1+fourier_order_out] # the constant coefficient
            end
            if ~isapprox(shift, zero(shift))
                to[1, cfc, :] .*= exp.(1im * shift * (-fourier_order_out:fourier_order_out))
            end
            # l select the variable in the monomial
            for l = 1:size(M1.mexp, 1)
                # multiply the exponent times
                for p = 1:M1.mexp[l, k]
                    # should not add to the previous result
                    res .= 0
                    @views FourierPolyMul!(res[1, :, :], to[1, :, :], X2[l, :, :], tab)
                    to .= res
                end
            end
            Xout[d, :, :] .+= to[1, index_map, :]
        end
    end
    return nothing
end

function QPFourierPolynomialSubstitute!(Mout::QPFourierPolynomial{dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder}, Xout,
    M1::QPFourierPolynomial, X1,
    M2::QPFourierPolynomial, X2; shift=0.0) where {dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder}
    tabs = QPFourierSubstituteTabs(Mout, Xout, M1, X1, M2, X2)
    return QPFourierPolynomialSubstitute!(Mout, Xout, M1, X1, M2, X2, tabs..., shift=shift)
end

function QPFourierDeriMul!(Mout::QPFourierPolynomial{dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder}, Xout,
    M1::QPFourierPolynomial, X1,
    M2::QPFourierPolynomial, X2) where {dim_out,dim_in,fourier_order_out,min_polyorder,max_polyorder}
    tab = mulTable(Mout.mexp, M1.mexp, M2.mexp)
    deri = zeros(eltype(Xout), size(X1, 2), size(X1, 3))
    Xout .= zero(eltype(Xout))
    for i = 1:dim_out
        # r the variable differentiated against
        for r = 1:dim_in
            deri .= zero(eltype(deri))
            for q in axes(X1, 3), k in axes(M1.D1col, 1)
                deri[M1.D1col[k, r], q] += X1[i, M1.D1row[k, r], q] * M1.D1val[k, r]
            end
            @views FourierPolyMul!(Xout[i, :, :], deri[:, :], X2[r, :, :], tab)
        end
    end
    nothing
end

function FourierWeights(fourier_order::Integer, theta::Vector)
    if eltype(theta) <: Complex
        out = zeros(eltype(theta), 2 * fourier_order + 1, length(theta))
    else
        out = zeros(Complex{eltype(theta)}, 2 * fourier_order + 1, length(theta))
    end
    for k = 1:length(theta)
        for f = -fourier_order:fourier_order
            out[fourier_order+1+f, k] = exp(1im * f * theta[k])
        end
    end
    return out
end

function FourierWeights(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, theta) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    #     out = zeros(ComplexF64, 2*fourier_order+1, length(theta))
    #     for k=1:length(theta)
    #         for f=-fourier_order:fourier_order
    #             out[fourier_order + 1 + f, k] = exp(1im*f*theta[k])
    #         end
    #     end
    return FourierWeights(fourier_order, theta)
end

# here theta is the weights of the fourier coefficients
function Eval(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, X, theta::Array{Complex{T},2}, data, L0=nothing) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,T}
    #     @show size(X), size(M.mexp), dim_out, dim_in, fourier_order, min_polyorder, max_polyorder
    MON = DenseMonomials(M.mexp, data)
    if L0 == nothing
        #         @show size(X), size(data), size(MON), size(theta)
        @tullio res[i, k] := X[i, p, q] * MON[p, k] * theta[q, k]
        return res
    else
        @tullio res = L0[i, k] * X[i, p, q] * MON[p, k] * theta[q, k]
        return res
    end
    nothing
end

function Eval(M::QPFourierPolynomial, X, theta::Array{U,1}, data::Array{T,2}, L0=nothing) where {U, T}
    return Eval(M, X, FourierWeights(M, theta), data, L0)
end

# Jacobian multiplied from the left
function JF_dx(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, X, theta, data, dx) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    to_MON = DenseMonomials(M.mexp, data)
    res = zeros(eltype(X), size(X, 1), size(data, 2))
    #     print("JF_dx:")
    #     @time 
    JF_dx_helper!(res, X, to_MON, theta, dx, M.D1row, M.D1col, M.D1val)
    return res
end

function QPFourierSubsTest()
    #     QPFourierPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder)
    M1 = QPFourierPolynomial(2, 5, 5, 0, 3)
    M2 = QPFourierPolynomial(5, 2, 5, 0, 3)
    Mout = QPFourierPolynomial(2, 2, 20, 0, 9)
    X1 = randn(M1)
    X2 = randn(M2)
    Xout = zero(Mout)

    omega = 1.0

    data = randn(2, 100) / 10
    theta = rand(100) * 2 * pi
    theta_shift = mod.(theta .+ omega, 2 * pi)
    w_shift = FourierWeights(M1, theta_shift)
    w = FourierWeights(M2, theta)
    wout = FourierWeights(Mout, theta)

    QPFourierPolynomialSubstitute!(Mout, Xout, M1, X1, M2, X2, shift=omega)
    res = Eval(Mout, Xout, wout, data) .- Eval(M1, X1, w_shift, Eval(M2, X2, w, data))
    @show abs.(res)
    @show maximum(abs.(res))
    nothing
end

#------------------------------------------------------------------------------------------------------------------------------------------
#
# quasi periodic vector values function
#
#------------------------------------------------------------------------------------------------------------------------------------------

struct QPConstant{dim_out,fourier_order,𝔽} <: AbstractManifold{𝔽}
    M::AbstractManifold{𝔽}
    R::AbstractRetractionMethod
    VT::AbstractVectorTransportMethod
end

@doc raw"""
    M = QPConstant(dim_out, fourier_order, field::AbstractNumbers=ℝ)
    
Creates a vector valued periodic function with `fourier_order` Fourier harmonics. The parameters are
* `dim_out` number of dimensions,
* `fourier_order` number of Fourier harmonics,
* `field` whether it is real valued `field=ℝ` or complex valued `field=ℂ`.
"""
function QPConstant(dim_out, fourier_order, field::AbstractNumbers=ℝ)
    M = Euclidean(dim_out, 2 * fourier_order + 1, field=field)
    R = ExponentialRetraction()
    VT = ParallelTransport()
    return QPConstant{dim_out,fourier_order,field}(M, R, VT)
end

function zero(M::QPConstant{dim_out,fourier_order,ℝ}) where {dim_out,fourier_order}
    return zeros(dim_out, 2 * fourier_order + 1)
end

function randn(M::QPConstant{dim_out,fourier_order,ℝ}, p=nothing) where {dim_out,fourier_order}
    return randn(dim_out, 2 * fourier_order + 1)
end

function thetaDerivative(M::QPConstant{dim_out,fourier_order,ℝ}, X, omega) where {dim_out,fourier_order}
    grid = getgrid(fourier_order)
    SH = differentialOperator(grid, omega)
    @tullio DX[i, j] := SH[j, k] * X[i, k]
    return DX
end

function makeCache(M::QPConstant{dim_out,fourier_order,ℝ}, X, theta::Array{T,2}, data=nothing) where {dim_out,fourier_order,T}
    return X * theta
end

function updateCache!(cache, M::QPConstant{dim_out,fourier_order,ℝ}, X, theta::Array{T,2}, data=nothing) where {dim_out,fourier_order,T}
    cache .= X * theta
    nothing
end

function Eval(M::QPConstant{dim_out,fourier_order,ℝ}, X, theta::Array{T,2}, data=nothing; cache=makeCache(M, X, theta, data)) where {dim_out,fourier_order,T}
    return cache
end

function InterpolationWeights(M::QPConstant{dim_out,fourier_order,field}, theta) where {dim_out,fourier_order,field}
    grid = getgrid(fourier_order)
    return fourierInterplate(grid, theta)
end

function Eval(M::QPConstant, X, theta::Array{T,1}, data=nothing) where {T}
    return Eval(M, X, InterpolationWeights(M, theta), data)
end

# vector valued
function Eval(M::QPConstant, X, theta::Number, data=nothing)
    return vec(Eval(M, X, InterpolationWeights(M, [theta])))
end

function L0_DF(M::QPConstant{dim_out,fourier_order,ℝ}, X, theta, data, L0; cache=nothing) where {dim_out,fourier_order}
    @tullio DX[i, q] := L0[i, k] * theta[q, k]
    return DX
end

function DF_dt(M::QPConstant, X, theta, data; dt, cache=nothing)
    return dt * theta
end

#------------------------------------------------------------------------------------------------------------------------------------------
#
# quasi periodic General Matrix and STIEFEL
#
#------------------------------------------------------------------------------------------------------------------------------------------

struct QPMatrix{dim_out,dim_in,fourier_order,tall,𝔽} <: AbstractManifold{𝔽}
    M::AbstractManifold{𝔽}
    R::AbstractRetractionMethod
    VT::AbstractVectorTransportMethod
end

# This is always tall, because it is a standard matrix 
function QPMatrix(dim_out, dim_in, fourier_order, field::AbstractNumbers=ℝ)
    R = ExponentialRetraction()
    VT = ParallelTransport()
    #     if dim_out >= dim_in
    # tall
    M = Euclidean(dim_out, dim_in, field=field)
    return QPMatrix{dim_out,dim_in,fourier_order,true,field}(M, R, VT)
    #     else
    #         # flat
    #         M = Euclidean(dim_in, dim_out, field=field)
    #         return QPMatrix{dim_out, dim_in, fourier_order, false, field}(M, R, VT)
    #     end
    #     nothing
end

function QPStiefel(dim_out, dim_in, fourier_order, field::AbstractNumbers=ℝ)
    R = PolarRetraction()
    VT = DifferentiatedRetractionVectorTransport{PolarRetraction}(PolarRetraction())
    if dim_out >= dim_in
        # tall
        M = Stiefel(dim_out, dim_in, field)
        return QPMatrix{dim_out,dim_in,fourier_order,true,field}(M, R, VT)
    else
        # flat
        M = Stiefel(dim_in, dim_out, field)
        return QPMatrix{dim_out,dim_in,fourier_order,false,field}(M, R, VT)
    end
    nothing
end

function zero(M::QPMatrix{dim_out,dim_in,fourier_order,true,ℝ}) where {dim_out,dim_in,fourier_order}
    return zeros(dim_out, dim_in, 2 * fourier_order + 1)
end

function zero(M::QPMatrix{dim_out,dim_in,fourier_order,false,ℝ}) where {dim_out,dim_in,fourier_order}
    return zeros(dim_in, dim_out, 2 * fourier_order + 1)
end

function randn(M::QPMatrix{dim_out,dim_in,fourier_order,tall,ℝ}) where {dim_out,dim_in,fourier_order,tall}
    q = zero(M)
    for k = 1:size(q, 3)
        @views q[:, :, k] .= rand(M.M)
    end
    return q
end

function randn(M::QPMatrix{dim_out,dim_in,fourier_order,tall,ℝ}, p) where {dim_out,dim_in,fourier_order,tall}
    q = zero(M)
    for k = 1:size(q, 3)
        @views q[:, :, k] .= rand(M.M; vector_at=p[:, :, k])
    end
    return q
end

# function randn(M::QPMatrix{dim_out, dim_in, fourier_order, false, ℝ}) where {dim_out, dim_in, fourier_order}
#     return randn(dim_in, dim_out, 2*fourier_order+1)
# end

function makeCache(M::QPMatrix{dim_out,dim_in,fourier_order,true,ℝ}, X, theta, data) where {dim_out,dim_in,fourier_order}
    @tullio res[i, k] := X[i, p, q] * data[p, k] * theta[q, k]
    return res
end

function makeCache(M::QPMatrix{dim_out,dim_in,fourier_order,false,ℝ}, X, theta, data) where {dim_out,dim_in,fourier_order}
    @tullio res[i, k] := X[p, i, q] * data[p, k] * theta[q, k]
    return res
end

function updateCache!(cache, M::QPMatrix{dim_out,dim_in,fourier_order,true,ℝ}, X, theta, data) where {dim_out,dim_in,fourier_order}
    @tullio cache[i, k] = X[i, p, q] * data[p, k] * theta[q, k]
    return nothing
end

function updateCache!(cache, M::QPMatrix{dim_out,dim_in,fourier_order,false,ℝ}, X, theta, data) where {dim_out,dim_in,fourier_order}
    @tullio cache[i, k] = X[p, i, q] * data[p, k] * theta[q, k]
    return nothing
end

function Eval(M::QPMatrix, X, theta, data; cache=makeCache(M, X, theta, data))
    return cache
end

function L0_DF(M::QPMatrix{dim_out,dim_in,fourier_order,true,ℝ}, X, theta, data, L0; cache=nothing) where {dim_out,dim_in,fourier_order}
    @tullio DX[i, p, q] := L0[i, k] * data[p, k] * theta[q, k]
    return DX
end

function L0_DF(M::QPMatrix{dim_out,dim_in,fourier_order,false,ℝ}, X, theta, data, L0; cache=nothing) where {dim_out,dim_in,fourier_order}
    @tullio DX[p, i, q] := L0[i, k] * data[p, k] * theta[q, k]
    return DX
end

function L0_JF(M::QPMatrix{dim_out,dim_in,fourier_order,true,ℝ}, X, theta, data, L0; cache=nothing) where {dim_out,dim_in,fourier_order}
    @tullio res[p, k] := L0[i, k] * X[i, p, q] * theta[q, k]
    return res
end

function DF_dt(M::QPMatrix{dim_out,dim_in,fourier_order,false,ℝ}, X, theta, data; dt, cache=nothing) where {dim_out,dim_in,fourier_order}
    @tullio res[i, k] := dt[p, i, q] * data[p, k] * theta[q, k]
    return res
end

function DF_dt(M::QPMatrix{dim_out,dim_in,fourier_order,true,ℝ}, X, theta, data; dt, cache=nothing) where {dim_out,dim_in,fourier_order}
    @tullio res[i, k] := dt[i, p, q] * data[p, k] * theta[q, k]
    return res
end

function JF(M::QPMatrix{dim_out,dim_in,fourier_order,true,ℝ}, X, theta, data; cache=nothing) where {dim_out,dim_in,fourier_order}
    @tullio res[i, p, k] := X[i, p, q] * theta[q, k]
    return res
end

function L0_HF(M::QPMatrix{dim_out,dim_in,fourier_order,true,ℝ}, X, theta, data, L0; cache=nothing) where {dim_out,dim_in,fourier_order}
    return zeros(eltype(X), dim_in, dim_in, size(data, 2))
end

function copyMost!(MD::QPMatrix, XD, MS::QPPolynomial, XS)
    XD .= GetLinearPart(MS, XS)
    return nothing
end

function copyMost!(MD::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, XD, MS::QPMatrix, XS) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    SetLinearPart!(MD, XD, XS)
    nothing
end

function copyMost!(MD::QPConstant, XD, MS::QPConstant, XS)
    XD .= XS
    return nothing
end

function copyMost!(MD::QPMatrix, XD, MS::QPMatrix, XS)
    XD .= XS
    return nothing
end

import ManifoldsBase.zero_vector!

function zero_vector!(M::QPConstant, X, p)
    return zero_vector!(M.M, X, p)
end

function zero_vector!(M::QPMatrix, X, p)
    for k = 1:size(X, 3)
        @views zero_vector!(M.M, X[:, :, k], p[:, :, k])
    end
    return X
end

function zero_vector!(M::QPPolynomial, X, p)
    return zero_vector!(M.M, X, p)
end

import ManifoldsBase.manifold_dimension

function manifold_dimension(M::QPConstant)
    return manifold_dimension(M.M)
end

function manifold_dimension(M::QPMatrix{dim_out,dim_in,fourier_order,tall,ℝ}) where {dim_out,dim_in,fourier_order,tall}
    return manifold_dimension(M.M) * (2 * fourier_order + 1)
end

function manifold_dimension(M::QPPolynomial)
    return manifold_dimension(M.M)
end

import ManifoldsBase.inner

function inner(M::QPConstant, p, X, Y)
    return inner(M.M, p, X, Y)
end

function inner(M::QPMatrix, p, X, Y)
    return sum(X .* Y)
end

function inner(M::QPPolynomial, p, X, Y)
    return inner(M.M, p, X, Y)
end

import ManifoldsBase.project!

function project!(M::QPConstant, Y, q, X)
    return project!(M.M, Y, q, X)
end

function project!(M::QPMatrix, Y, q, X)
    #     println("QPMatrix project", typeof(M.M))
    for k = 1:size(X, 3)
        @views project!(M.M, Y[:, :, k], q[:, :, k], X[:, :, k])
    end
    return Y
end

function project!(M::QPPolynomial, Y, q, X)
    return project!(M.M, Y, q, X)
end

import ManifoldsBase.retract!

function retract!(M::QPConstant, q, p, X, method::AbstractRetractionMethod=M.R)
    return retract!(M.M, q, p, X, method)
end

function retract!(M::QPConstant, q, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    return retract!(M.M, q, p, X, t, method)
end

function retract!(M::QPMatrix, q, p, X, method::AbstractRetractionMethod=M.R)
    #     println("QPMatrix retract! ", typeof(M.M), typeof(method))
    for k = 1:size(X, 3)
        @views retract!(M.M, q[:, :, k], p[:, :, k], X[:, :, k], method)
    end
    return q
end

function retract!(M::QPMatrix, q, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    #     println("QPMatrix retract! t ", typeof(M.M), typeof(method))
    for k = 1:size(X, 3)
        @views retract!(M.M, q[:, :, k], p[:, :, k], X[:, :, k], t, method)
    end
    return q
end

function retract!(M::QPPolynomial, q, p, X, method::AbstractRetractionMethod=M.R)
    return retract!(M.M, q, p, X, method)
end

function retract!(M::QPPolynomial, q, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    return retract!(M.M, q, p, X, t, method)
end

import ManifoldsBase.retract

function retract(M::QPConstant, p, X, method::AbstractRetractionMethod=M.R)
    return retract(M.M, p, X, method)
end

function retract(M::QPConstant, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    return retract(M.M, p, X, t, method)
end

function retract(M::QPMatrix, p, X, method::AbstractRetractionMethod=M.R)
    #     println("QPMatrix retract ", typeof(M.M), typeof(method))
    q = zero(p)
    for k = 1:size(X, 3)
        @views q[:, :, k] .= retract(M.M, p[:, :, k], X[:, :, k], method)
        #         @views q[:,:,k] .= retract(M.M, p[:,:,k], X[:,:,k], PolarRetraction())
    end
    return q
end

function retract(M::QPMatrix, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    #     println("QPMatrix retract t ", typeof(M.M), typeof(method))
    q = zero(p)
    for k = 1:size(X, 3)
        @views q[:, :, k] .= retract(M.M, p[:, :, k], X[:, :, k], t, method)
    end
    return q
end

function retract(M::QPPolynomial, p, X, method::AbstractRetractionMethod=M.R)
    return retract(M.M, p, X, method)
end

function retract(M::QPPolynomial, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    return retract(M.M, p, X, t, method)
end

import ManifoldsBase.vector_transport_to!

function vector_transport_to!(M::QPConstant, Y, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to!(M::QPMatrix, Y, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    for k = 1:size(X, 3)
        @views vector_transport_to!(M.M, Y[:, :, k], p[:, :, k], X[:, :, k], q[:, :, k], method)
    end
    return Y
end

function vector_transport_to!(M::QPPolynomial, Y, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

import ManifoldsBase.vector_transport_to

function vector_transport_to(M::QPConstant, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to(M.M, p, X, q, method)
end

function vector_transport_to(M::QPMatrix, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    Y = zero(X)
    for k = 1:size(X, 3)
        @views Y[:, :, k] .= vector_transport_to(M.M, p[:, :, k], X[:, :, k], q[:, :, k], method)
    end
    return Y
end

function vector_transport_to(M::QPPolynomial, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to(M.M, p, X, q, method)
end

# only occurs here, due to BFGS Hessian Approximation
import ManifoldsBase.get_coordinates
import ManifoldsBase.get_coordinates!

function get_coordinates(M::QPConstant, p, X, B::AbstractBasis)
    return get_coordinates(M.M, p, X, B)
end

function get_coordinates(M::QPMatrix, p, X, B::AbstractBasis)
    V = get_coordinates(M.M, p[:, :, 1], X[:, :, 1], B)
    Y = zeros(eltype(V), size(V)..., size(X, 3))
    Y[:, 1] .= V
    for k = 2:size(X, 3)
        Y[:, k] .= get_coordinates(M.M, p[:, :, k], X[:, :, k], B)
    end
    return vec(Y)
end

function get_coordinates(M::QPPolynomial, p, X, B::AbstractBasis)
    return get_coordinates(M.M, p, X, B)
end

import ManifoldsBase.get_vector
import ManifoldsBase.get_vector!

function get_vector(M::QPConstant, p, c, B::AbstractBasis)
    return get_vector(M.M, p, c, B)
end

function get_vector(M::QPMatrix, p, c0, B::AbstractBasis)
    c = reshape(c0, :, size(p, 3))
    V = get_vector(M.M, p[:, :, 1], c[:, 1], B)
    Y = zeros(eltype(V), size(V)..., size(p, 3))
    Y[:, :, 1] .= V
    for k = 2:size(p, 3)
        Y[:, :, k] .= get_vector(M.M, p[:, :, k], c[:, k], B)
    end
    return Y
end

function get_vector(M::QPPolynomial, p, c, B::AbstractBasis)
    return get_vector(M.M, p, c, B)
end

# Hessian projection

function SymPart(V)
    return (transpose(V) + V) / 2.0
end

function HessianProjection(M::Stiefel, X, grad, HessV, V)
    return project(M, X, HessV - V * SymPart(transpose(X) * grad))
end

function HessianProjection(M::QPMatrix, X, grad, HessV, V)
    Y = zero(X)
    for k = 1:size(X, 3)
        @views project!(M.M, Y[:, :, k], X[:, :, k], HessV[:, :, k] - V[:, :, k] * SymPart(transpose(X[:, :, k]) * grad[:, :, k]))
    end
    return Y
end

function HessianProjection(M::Euclidean, X, grad, HessV, V)
    return HessV
end

function HessianProjection(M::ProductManifold, X, grad, HessV, V)
    return ArrayPartition(map(HessianProjection, M.manifolds, X.x, grad.x, HessV.x, V.x))
end

#-----------------------------------------------------------------------------------
# METHODS THAT REQUIRE OTHER CLASSES
#-----------------------------------------------------------------------------------

# the full Jacobian around a torus K
# it is requred to find a torus in a map
function Jacobian(M::QPPolynomial{dim_out,dim_out,fourier_order,min_polyorder,max_polyorder,ℝ}, X, MK::QPConstant{dim_out,fourier_order,ℝ}, XK) where {dim_out,fourier_order,min_polyorder,max_polyorder}
    to_exp = DensePolyExponents(dim_out, max(0, max_polyorder - 1); min_order=max(0, min_polyorder - 1))
    D1row, D1col, D1val = FirstDeriMatrices(to_exp, M.mexp)
    MON = DenseMonomials(to_exp, XK)
    MJ = QPMatrix(dim_out, dim_out, fourier_order)
    XJ = zero(MJ)
    for r = 1:size(D1row, 2), q = 1:size(X, 3), k = 1:size(D1row, 1)
        XJ[:, r, q] .+= X[:, D1row[k, r], q] * D1val[k, r] * MON[D1col[k, r], q]
    end
    return MJ, XJ
end

function testQP()
end
