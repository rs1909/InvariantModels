function findTorus(A, b, omega)
    ndim = size(A,1)
    tdim = size(A,3)
    fourier_order = div(tdim-1,2)
    
    # need to solve K(t+omega) = A(t) K(t) + b(t)
    #   S_{-omega} K = A
    grid = getgrid(fourier_order)
    S = shiftOperator(grid, -omega)
    
    # i -> ndim
    # j -> tdim
    # k -> ndim
    # l -> tdim
    In = Diagonal(I,ndim)
    It = Diagonal(I,tdim)
    @tullio AA[i,j,k,l] := In[i,k] * S[j,l] - A[i,k,l] * It[j,l]
    AAhat = reshape(AA, ndim*tdim, ndim*tdim)
    K0 = AAhat \ vec(b)
    K = reshape(K0, size(b)...)
    return K, S
end

function findLinearModelHelper(dataX, thetaX, dataY, thetaY, omega)
    ndim = size(dataX,1)
    tdim = size(thetaX,1)
    fourier_order = div(tdim-1,2)
    #
    @tullio xot_3d[i,j,k] := dataX[i,k] * thetaX[j,k]
    xot = reshape(xot_3d,:, size(xot_3d,3))
    xhat = zeros(eltype(xot), size(xot,1) + size(thetaX,1), size(xot,2))
    xhat[1:size(xot,1),:] .= xot
    xhat[size(xot,1)+1:end,:] .= thetaX

    scale = ones(1,size(xhat,2))
    # this is where we can scale
    local A, b, MK, S
    K = zeros(eltype(dataX), ndim, tdim)
    for k=1:20
        XX = (xhat .* scale) * transpose(xhat)
        YY = (dataY .* scale) * transpose(xhat)
        Ahat = YY / XX
        
        A = reshape(Ahat[:,1:size(xot,1)], size(Ahat,1), size(Ahat,1), :)
        b = Ahat[:,size(xot,1)+1:end]
        # residual of fitting
        @tullio res1[j,k] := A[j,p,q] * dataX[p,k] * thetaX[q,k]
        @tullio res2[j,k] := b[j,q] * thetaX[q,k]
        res = res1 + res2 - dataY
        
        K0, S = findTorus(A, b, omega)
        MK = QPConstant(ndim, fourier_order)
        err = norm(K - K0)
        if err < 20*eps(eltype(K))
            break
        end
        K .= K0
        println("E = ", err, " R = ", norm(res))
        scale .= one(eltype(scale)) ./ (sum((dataX .- K * thetaX) .^ 2, dims=1) .+ 0.1^2)
    end
    
    # residual of torus fitting
    println("Residuals of finding a torus and nearby linear dynamics.")
    @tullio KX[j,k] := K[j,p] * thetaX[p,k]
    @tullio KY[j,k] := K[j,p] * thetaY[p,k]
    @tullio AKX[j,k] := A[j,p,q] * KX[p,k] * thetaX[q,k]
    @tullio bX[j,k] := b[j,p] * thetaX[p,k]
    @show norm(KY - AKX - bX)
    @tullio Ksh[j,k] := K[j,p] * S[k,p]
    @tullio AK[j,k] := A[j,p,k] * K[p,k]
    @show norm(Ksh - AK - b)
    return A, b, MK, K
end

@doc raw"""
    findLinearModel(dataX, thetaX, dataY, thetaY, omega; ratio=0.5, steps=3)
    
The input arguments arguments
* `dataX` is matrix ``\boldsymbol{X}``. Each column corresponts to a data point.
* `dataY` is matrix ``\boldsymbol{Y}``.
* `thetaX` is matrix ``\boldsymbol{\Theta}_x``
* `thetaY` is matrix ``\boldsymbol{\Theta}_y``
* `ratio` the ratio of data to be kept at the final step
* `steps` the number of steps to take to find 

Finds a linear model in the form of
```math
\begin{aligned}
\boldsymbol{x}_{k+1} &= \boldsymbol{A}(\theta_k) \boldsymbol{x}_k + \boldsymbol{b}(\theta_k) \\
\theta_{k+1} &= \theta_k + \omega
\end{aligned}
```
and calculates the invariant torus represented by ``\boldsymbol{K}`` from the equation
```math
\boldsymbol{K}(\theta+\omega) = \boldsymbol{A}(\theta) \boldsymbol{K}(\theta) + \boldsymbol{b}(\theta)
```

The output is 
```
A, b, MK, XK
```
* `A` is a 3-index array representing ``\boldsymbol{A}(\theta)``
* `b` is a 2-index array representing ``\boldsymbol{b}(\theta)``
* `MK`, `XK` represent to torus parametrisation ``K``
"""
function findLinearModel(dataX, thetaX, dataY, thetaY, omega; ratio=0.5, steps=3)
    ratio_step = ratio ^ (1/steps)
    deviation = deepcopy(dataX)
    local A, b, MK, XK
    for k in 1:steps
        prm = sortperm(vec(sqrt.(sum(deviation .^ 2, dims=1))))
        id = sort(prm[1:Int(round(min(1, ratio_step ^ k) * length(prm)))])
        A, b, MK, XK = findLinearModelHelper(dataX[:, id], thetaX[:, id], dataY[:, id], thetaY[:, id], omega)
        deviation .= dataX - XK * thetaX
    end
    return A, b, MK, XK
end
