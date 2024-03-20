
# the function is phix,y) = exp(y/(C*(-D^2 - S*x + y)))*(1 + 1/(F^2 + x^P + y^P))
# D: size of cut-off
# C: power of the EXP function : how flat is it at the top
# S: The slope of the cone
# F: 1/(F^2 + x^P + y^P) factor
# P: the exponent above
# norm_beta: defines the threshold for the quasi-L1 norm. 
# the global foliation

const toProfile = false
const toDynScale = false

const bumpU_D = 1024 # 2^-4
const bumpU_C = 2
const bumpU_S = 1 # for the rest
const bumpU_F = 2^-10 #2^-8
const bumpU_P = 1
# The local foliation
const bumpV_D = 1024 # 2^-4 # for all the rest though
const bumpV_C = 2
const bumpV_S = 1 # for the rest
const bumpV_F = 2^-10 # 2^-8
const bumpV_P = 1
const norm_beta = 2^-12 # for one-mass # 1 for the rest

# Static scaling 1 + 1/(scale_epsilon ^ 2 + (X - Xstar) .^ 2)
const scale_epsilon = 2^-8

macro profile(ex)
    if toProfile
        return :($(esc(ex)))
    end
end

macro dynscale(ex)
    if toDynScale
        return :($(esc(ex)))
    end
end

macro staticscale(ex)
    if !toDynScale
        return :($(esc(ex)))
    end
end

# it contains the location of the torus with respect to the (transformed) data: MK, XK
struct QPCombinedFoliation{dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order,𝔽} <: AbstractManifold{𝔽}
    M::ProductManifold
    R::ProductRetraction
    VT::ProductVectorTransport
    MK::QPConstant{dim_data,fourier_order,𝔽}
    XK
end

@doc raw"""
    copyMost!(MCFD::QPCombinedFoliation, XCFD, MCFS::QPCombinedFoliation, XCFS)
    
Make a best-effort attempt to copy the combined foliation `MCFS`, `XCFS` into `MCFD`, `XCFD`.
"""
function copyMost!(MCFD::QPCombinedFoliation, XCFD, MCFS::QPCombinedFoliation, XCFS)
    map(copyMost!, MCFD.M.manifolds, XCFD.x, MCFS.M.manifolds, XCFS.x)
    nothing
end

const part_R = 1
const part_U0 = 2
const part_U1 = 3
const part_Unl = 4
const part_S = 5
const part_V0 = 6
const part_V1 = 7
const part_Vnl = 8

function QPC_R_manifold(M::QPCombinedFoliation)
    return M.M.manifolds[part_R]
end

function QPC_U0_manifold(M::QPCombinedFoliation)
    return M.M.manifolds[part_U0]
end

function QPC_U1_manifold(M::QPCombinedFoliation)
    return M.M.manifolds[part_U1]
end

function QPC_Unl_manifold(M::QPCombinedFoliation)
    return M.M.manifolds[part_Unl]
end

function QPC_S_manifold(M::QPCombinedFoliation)
    return M.M.manifolds[part_S]
end

function QPC_V0_manifold(M::QPCombinedFoliation)
    return M.M.manifolds[part_V0]
end

function QPC_V1_manifold(M::QPCombinedFoliation)
    return M.M.manifolds[part_V1]
end

function QPC_Vnl_manifold(M::QPCombinedFoliation)
    return M.M.manifolds[part_Vnl]
end

function QPC_R_point(X::ArrayPartition)
    return X.x[part_R]
end

function QPC_U0_point(X::ArrayPartition)
    return X.x[part_U0]
end

function QPC_U1_point(X::ArrayPartition)
    return X.x[part_U1]
end

function QPC_Unl_point(X::ArrayPartition)
    return X.x[part_Unl]
end

function QPC_S_point(X::ArrayPartition)
    return X.x[part_S]
end

function QPC_V0_point(X::ArrayPartition)
    return X.x[part_V0]
end

function QPC_V1_point(X::ArrayPartition)
    return X.x[part_V1]
end

function QPC_Vnl_point(X::ArrayPartition)
    return X.x[part_Vnl]
end

# this should also update cache at the same time
function Set_parts!(X::ArrayPartition, Xp, sel)
    if length(sel) == 1
        # S, U0, U1, B, V0, V1
        X.x[sel[1]] .= Xp
    else
        # Unl, Vnl
        X.x[sel[1]].x[sel[2]].x[sel[3]] .= Xp
    end
    nothing
end

function Get_parts(X::ArrayPartition, sel)
    if length(sel) == 1
        # S, U0, U1, B, V0, V1
        return X.x[sel[1]]
    else
        # Unl, Vnl
        return X.x[sel[1]].x[sel[2]].x[sel[3]]
    end
    return zeros(0, 0)
end

function Get_part_size(X::ArrayPartition, sel)
    if length(sel) == 1
        # S, U0, U1, B, V0, V1
        return size(X.x[sel[1]])
    else
        # Unl, Vnl
        return size(X.x[sel[1]].x[sel[2]].x[sel[3]])
    end
    return (0, 0)
end

function Get_MRVT(M::QPCombinedFoliation, sel)
    if length(sel) == 1
        # S, U0, U1, B, V0, V1
        return M.M.manifolds[sel[1]], M.R.retractions[sel[1]], M.VT.methods[sel[1]]
    else
        # Unl, Vnl
        return M.M.manifolds[sel[1]].M.manifolds[sel[2]].M.manifolds[sel[3]], M.M.manifolds[sel[1]].R.retractions[sel[2]].retractions[sel[3]], M.M.manifolds[sel[1]].VT.methods[sel[2]].methods[sel[3]]
    end
    return M
end

# this is a very general algorithm that goes throuh all the bodes of X and wraps around at the end
function nextSel(X::ArrayPartition, sel)
    function climbTree(X::ArrayPartition, sel)
        if isempty(sel)
            return X
        else
            return climbTree(X.x[sel[1]], sel[2:end])
        end
        nothing
    end

    # X relates to sel[end]
    function completeTree(X::ArrayPartition, sel)
        if hasproperty(X.x[sel[end]], :x)
            #         println("node ", sel)
            return completeTree(X.x[sel[end]], (sel..., 1))
        else
            #         println("leaf ", sel)
            return sel
        end
        nothing
    end

    # find the first non-end node as coming back down the tree
    # returns empty if at the very end
    function backTree(X::ArrayPartition, sel)
        if isempty(sel)
            return X, ()
        end
        a = climbTree(X, sel[1:end-1])
        if isa(a, ArrayPartition)
            if sel[end] < length(a.x)
                return a, sel
            else
                return backTree(X::ArrayPartition, sel[1:end-1])
            end
        else
            println("backTree: Something is very wrong")
            return a, (-1,)
        end
        nothing
    end

    a, selp = backTree(X, sel)
    if isempty(selp)
        return completeTree(X, (1,))
    else
        return completeTree(a, (selp[1:end-1]..., selp[end] + 1))
    end
    nothing
end

function countParts(X::ArrayPartition)
    sel = Array{Any,1}(undef, 1)
    sel[1] = (1,)
    count = 1
    while true
        sel[1] = nextSel(X, sel[1])
        if sel[1] == (1,)
            return count
        else
            count = count + 1
        end
    end
    return count
end

# exagerates S and U_0 (constant part of U)
function maximumNorm(X::ArrayPartition)
    sel = Array{Any,1}(undef, 1)
    sel[1] = (1,)
    selM = deepcopy(sel)
    valM = 0.0
    while true
        #         val = norm(Get_parts(X, sel[1]))
        # if the constant part, increase the importance
        # L_inf norm
        val = maximum(abs.(Get_parts(X, sel[1])))
        if val > valM
            selM .= sel
            valM = val
        end
        # exit 
        sel[1] = nextSel(X, sel[1])
        if sel[1] == (1,)
            return selM[1], valM
        end
    end
    return selM[1], valM
end

function zero_vector!(M::QPCombinedFoliation, X, p)
    return zero_vector!(M.M, X, p)
end

function manifold_dimension(M::QPCombinedFoliation)
    return manifold_dimension(M.M)
end

function inner(M::QPCombinedFoliation, p, X, Y)
    return inner(M.M, p, X, Y)
end

function project!(M::QPCombinedFoliation, Y, q, X)
    return project!(M.M, Y, q, X)
end

function retract!(M::QPCombinedFoliation, q, p, X, alpha::Number, method::AbstractRetractionMethod=M.R)
    return retract!(M.M, q, p, X, alpha, method)
end

function retract(M::QPCombinedFoliation, p, X, alpha::Number, method::AbstractRetractionMethod=M.R)
    return retract(M.M, p, X, alpha, method)
end

function vector_transport_to!(M::QPCombinedFoliation, Y, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::QPCombinedFoliation, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to(M.M, p, X, q, method)
end

function get_coordinates(M::QPCombinedFoliation, p, X, B::AbstractBasis)
    return get_coordinates(M.M, p, X, B)
end

function get_vector(M::QPCombinedFoliation, p, c, B::AbstractBasis)
    return get_vector(M.M, p, c, B)
end

function zero(M::QPCombinedFoliation)
    return ArrayPartition(map(zero, M.M.manifolds)...)
end

function randn(M::QPCombinedFoliation)
    return ArrayPartition(map(randn, M.M.manifolds)...)
end

# S1 : linear part of ROM
# U0 : constant part of the submersion
# U1 : linear part of the submersion, must be orthogonal
# V0 : constant part of the local submersion
# V1 : linear part of the local submersion, must be orthogonal
@doc raw"""
    QPCombinedFoliation(dim_data, dim_latent, fourier_order, R_order, U_order, S_order, V_order, R1, S1, MK, XK; sparse=false)
    
Creates the data structures to store a pair of foliations that might be interconnected through scaling. The invariance equations represented by a Combined Foliation are
```math
\begin{aligned}
\boldsymbol{R}\left(\boldsymbol{U}\left(\boldsymbol{x},\boldsymbol{\theta}\right),\boldsymbol{\theta}\right) &= \boldsymbol{U}\left(\boldsymbol{F}\left(\boldsymbol{x},\boldsymbol{\theta}\right),\boldsymbol{\theta}+\boldsymbol{\omega}\right)\\
\boldsymbol{S}\left(\boldsymbol{V}\left(\boldsymbol{x},\boldsymbol{\theta}\right),\boldsymbol{\theta}\right) &= \boldsymbol{V}\left(\boldsymbol{F}\left(\boldsymbol{x},\boldsymbol{\theta}\right),\boldsymbol{\theta}+\boldsymbol{\omega}\right)
\end{aligned}
```

The input parameters are
* `dim_data` system dimension
* `dim_latent` the dimension of the reduced order model
* `fourier_order `number of Fourier harmonics used in discretisation
* `R_order` the polynomial order of function ``\boldsymbol{R}``
* `U_order` the polynomial order of function ``\boldsymbol{U}``
* `S_order` the polynomial order of function ``\boldsymbol{S}``
* `V_order` the polynomial order of function ``\boldsymbol{V}``
* `R1` the initial linear part of function ``\boldsymbol{R}``
* `S1` the initial linear part of function ``\boldsymbol{S}``
* `MK`, `XK` the invariant torus
* `sparse` use compressed polynomials (default = `false`)

The output is the data structure
```
MCF, XCF
```
"""
function QPCombinedFoliation(dim_data, dim_latent, fourier_order, R_order, U_order, S_order, V_order, R1, S1, MK, XK; sparse=false)
    # R
    MR = QPPolynomial(dim_latent, dim_latent, fourier_order, 1, R_order)
    XR = zero(MR)
    SetLinearPart!(MR, XR, R1)
    # U0
    MU0 = QPConstant(dim_latent, fourier_order)
    XU0 = zero(MU0)
    # U1
    #     MU1 = QPMatrix(dim_latent, dim_data - dim_latent, fourier_order)
    MU1 = QPStiefel(dim_latent, dim_data, fourier_order)
    #     MU1 = QPMatrix(dim_latent, dim_data, fourier_order) # just for testing gradient
    XU1 = zero(MU1)
    XU1[1:dim_latent, :, :] .= Array(I, dim_latent, dim_latent)
    #     XU1[:,1:dim_latent,:] .= Array(I,dim_latent,dim_latent)
    # Unl
    if sparse
        # compressed polynomial
        MUnl = QPCompressedPolynomial(dim_latent, dim_data, (dim_latent+1):dim_data, fourier_order, 2, U_order; node_rank=4)
    else
        # dense polynomial
        MUnl = QPPolynomial(dim_latent, dim_data, fourier_order, 2, U_order; perp_idx=(dim_latent+1):dim_data)
    end
    XUnl = zero(MUnl)
    # local foliation
    # B
    if S_order == 1
        MS = QPMatrix(dim_data - dim_latent, dim_data - dim_latent, fourier_order)
        XS = zero(MS)
        XS .= S1
    else
        MS = QPPolynomial(dim_data - dim_latent, dim_data - dim_latent, fourier_order, 1, S_order)
        XS = zero(MS)
        SetLinearPart!(MS, XS, S1)
    end
    # V0
    MV0 = QPConstant(dim_data - dim_latent, fourier_order)
    XV0 = zero(MV0)
    # V1
    #     MV1 = QPMatrix(dim_data - dim_latent, dim_latent, fourier_order)
    MV1 = QPStiefel(dim_data - dim_latent, dim_data, fourier_order)
    #     MV1 = QPMatrix(dim_data - dim_latent, dim_data, fourier_order)
    XV1 = zero(MV1)
    XV1[dim_latent+1:end, :, :] .= Array(I, dim_data - dim_latent, dim_data - dim_latent)
    #     XV1[:,dim_latent+1:end,:] .= Array(I,dim_data - dim_latent,dim_data - dim_latent)
    # Vnl
    if sparse
        MVnl = QPCompressedPolynomial(dim_data - dim_latent, dim_data, 1:dim_latent, fourier_order, 2, V_order; node_rank=4)
    else
        MVnl = QPPolynomial(dim_data - dim_latent, dim_data, fourier_order, 2, V_order; perp_idx=1:dim_latent)
    end
    XVnl = zero(MVnl)

    M = ProductManifold(MR, MU0, MU1, MUnl, MS, MV0, MV1, MVnl)
    R = ProductRetraction(MR.R, MU0.R, MU1.R, MUnl.R, MS.R, MV0.R, MV1.R, MVnl.R)
    VT = ProductVectorTransport(MR.VT, MU0.VT, MU1.VT, MUnl.VT, MS.VT, MV0.VT, MV1.VT, MVnl.VT)

    MM = QPCombinedFoliation{dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order,ℝ}(M, R, VT, MK, XK)
    XM = ArrayPartition(XR, XU0, XU1, XUnl, XS, XV0, XV1, XVnl)
    return MM, XM
end

@doc raw"""
    SetTorus!(MCF::QPCombinedFoliation, MK::QPConstant, XK)
    
Set the location of the torus `MK`, `XK` for the combined foliation `MCF`.
"""
function SetTorus!(MCF::QPCombinedFoliation{dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order,ℝ}, MK::QPConstant{dim_data,fourier_order,ℝ}, XK) where {dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order}
    MCF.XK .= XK
    nothing
end

@doc raw"""
    bincuts(long_amp, trans_amps, max_amp, max_points, nbins; exponent=1)

This function is used to filter out data points that are far from an invariant manifold. 
    
* `long_amps` vector of amplitudes along the invariant manifold
* `trans_amps` vector of distances from the invariant manifold
* `max_amp` discard points that have greater amplitude than `max_amp`
* `max_points` the maximum number of points to return
* `nbins` the number of bins along which the distribution is made uniform
* `exponent` used to calculate bin sizes. The bins fall between the nodes of `range(0, max_amp, length=nbins + 1) .^ exponent`.

Returns the indices of `long_amps` that create a uniform distribution with respect to the bins and only contain the smallest `trans_amps` values from those bins.
"""
function bincuts(long_amp, trans_amps, max_amp, max_points, nbins; exponent=1)
    ma = maximum(long_amp)
    # maf is the maximum amplitude to be considered
    maf = min(ma, max_amp)
    # bounds contains the bin boundaries (starting with zero)
    bounds = maf * (range(0, 1, length=nbins + 1) .^ exponent)
    # bins will contain the indices for the amplitudes
    bins = Array{Array{Int,1},1}(undef, nbins)
    for k in eachindex(bins)
        bins[k] = findall((long_amp .>= bounds[k]) .&& (long_amp .< bounds[k+1]))
    end
    # ll has the lengths of each bin
    ll = length.(bins)
    llp = sortperm(ll, rev=true)
    # lls has the biggest bin first
    lls = ll[llp]
    # finding out home many element should be in each box
    let k = 1
        to_rem = sum(lls) - max_points
        @show to_rem
        while to_rem > 0 && k < nbins
            # how much can be removed just by chopping
            rem = sum(lls[1:k] .- lls[k+1])
            if rem <= to_rem && (k + 1) < nbins
                lls[1:k] .= lls[k+1]
                to_rem -= rem
            else
                drem = div(to_rem, k + 1)
                lls[1:k+1] .-= drem
                to_rem -= (k + 1) * drem
            end
            k += 1
        end
    end
    # putting back the relevant numbers
    ll = lls[invperm(llp)]
    for k in eachindex(bins)
        bp = sortperm(trans_amps[bins[k]])
        bins[k] = bins[k][bp[1:ll[k]]]
        #         bins[k] = bins[k][randperm(length(bins[k]))[1:ll[k]]]
    end
    indices = sort(vcat(bins...))
    return indices
end

# input: 
#       K -> torus
#       A -> linear map about torus
#       omega -> rotation
#       thetaX, dataX, thetaY, dataY -> data input
#       sel -> which 
# output: 
#       dataUX, dataVX, dataUY, dataVY -> data projected into subspace
#       R1, S1 -> linear part of the reduced model, although a full best-fit could be provided from the data
# Issues:
#       does not return the coordinate system, so we will not know how the data was transformed
# TODO needs to include caping the data with respect to non-orthogonal coordinate system
@doc raw"""
    QPPreProcess(K, A, omega, thetaX, dataX, thetaY, dataY, sel; Tstep=1.0, maxAmp=Inf, data_ratio=1.0, preId=[])

This function is called after a linear model has been identified using [`findLinearModel`](@ref). The following calculations are carried out
1. decompose the linear dynamics into invariant vector bundles.
2. project the data onto two invariant vector bundles, one that corresponds to the spectral circles selected by `sel` and the the remaining bundle. In this coordinate system the approximate linear model represented by `A` becomes block-diagonal.
3. 

The input arguments arguments
* `K` approximate invariant torus
* `A` approximate linear map about the torus
* `dataX` is matrix ``\boldsymbol{X}``. Each column corresponts to a data point.
* `dataY` is matrix ``\boldsymbol{Y}``.
* `thetaX` is matrix ``\boldsymbol{\Theta}_x`` These are the interpolations vectors for Fourier collocation corresponding to ``\theta_k``.
* `thetaY` is matrix ``\boldsymbol{\Theta}_y``
* `sel` indicates the selected vector bundles.
* `Tstep` is the sampling period ``\Delta t``
* `maxAmp` is the maximum amplitude of the data along the selected vector bundle
* `data_ratio` ``\in [0,1]`` the proportion of the data to be kept and the rest is removed so that it is uniformly distributed along in the neighbourhood of the selected vector bundles.
* `preId` if non-empty, this replaces the data filtering process.

The output is
```
thetaTX, dataTX, thetaTY, dataTY, dataScale, id, R1, S1, W1
```
* `thetaTX, dataTX, thetaTY, dataTY` are the filtered and scaled data. The approximate invariant torus is subtracted and the data is transformed into the coordiante system of the approximate vector bundles calculated from `A`.
* `dataScale` is the scaling factor that relates the input and output `dataX[:,id] = dataScale * dataTX`.
* `id` the indices of the input that are brought to the output
* `R1` is the invariant part of the the linear system `A` that is invariant under the selected `sel` vector bundles
* `S1` is the invariant part of the the linear system `A` that is invariant under the not selected (`setdiff(1:data_dim, sel)`) vector bundles
* `W1` the inverse transformation that brings that data back into the physical coordinate system.
"""
function QPPreProcess(K, A, omega, thetaX, dataX, thetaY, dataY, sel; Tstep=1.0, maxAmp=Inf, data_ratio=1.0, preId=[], dataScale = 0.0)
    SEL = vec(sel)
    fourier_order = div(size(K, 2) - 1, 2)
    @assert size(K, 2) == 2 * fourier_order + 1 "Incorrect grid size"
    dim_data = size(K, 1)
    dim_latent = length(SEL)
    # input A, b : F(x,t) = A(t) x + b(t)
    # so create a polynomial with A, b
    MF = QPPolynomial(dim_data, dim_data, fourier_order, 1, 1, ℝ)
    XF = zero(MF)
    SetLinearPart!(MF, XF, A)
    XU, XW, XUre, XWre, XUreOrth, XWreOrth, Lambda = vectorBundles(fourier_order, A, omega; sel=SEL, ODE=false, Tstep=Tstep)

    # transformation that leads to autonomous linear part
#     U1aut = XU[SEL, :, :]
#     W1aut = XW[:, SEL, :]

    U1 = XUreOrth[SEL, :, :]
    V1 = XUreOrth[setdiff(1:dim_data, SEL), :, :]
    WU1 = XWreOrth[:, SEL, :]
    
    W1 = similar(XWreOrth)
    W1[:,1:dim_latent,:] .= XWreOrth[:, SEL, :]
    W1[:,dim_latent+1:end,:] .= XWreOrth[:, setdiff(1:dim_data, SEL), :]

    @tullio pp[i,j,k] := U1[i,p,k] * W1[p,j,k]
    @tullio qq[i,j,k] := V1[i,p,k] * W1[p,j,k]
    @show pp
    @show qq

    MWs, XWs, MUs, XUs, MUFWs, XUFWs = bundleTransform(MF, XF, XUreOrth, XWreOrth, omega, SEL)
    MWc, XWc, MUc, XUc, MUFWc, XUFWc = bundleTransform(MF, XF, XUreOrth, XWreOrth, omega, setdiff(1:dim_data, SEL))
    # discards MWs, XWs, MUs, XUs and MWc, XWc, MUc, XUc as they are the same as XUre, XWre ...
    R1 = GetLinearPart(MUFWs, XUFWs)
    S1 = GetLinearPart(MUFWc, XUFWc)

    dX = dataX - (K * thetaX)
    dY = dataY - (K * thetaY)
    @tullio dataUX[i, k] := U1[i, j, p] * dX[j, k] * thetaX[p, k]
    @tullio dataUY[i, k] := U1[i, j, p] * dY[j, k] * thetaY[p, k]
    @tullio dataVX[i, k] := V1[i, j, p] * dX[j, k] * thetaX[p, k]
    @tullio dataVY[i, k] := V1[i, j, p] * dY[j, k] * thetaY[p, k]
    long_amps = vec(sqrt.(sum(dataUX .^ 2, dims=1)))
    trans_amps = vec(sqrt.(sum(dataVX .^ 2, dims=1)))
    
    @tullio P1[i, j, p] := WU1[i, q, p] * U1[q, j, p]
    @tullio dataXproj[i, k] := P1[i, j, p] * dX[j, k] * thetaX[p, k]

    @show mx_proj = sqrt(maximum(vec(sum(real.(dataXproj .* conj.(dataXproj)), dims=1))))
    @show mx_long = maximum(trans_amps)

    if isempty(preId)
        id = bincuts(long_amps, trans_amps, mx_long * (maxAmp / mx_proj), Int(round(size(dataX, 2) * data_ratio)), 30)
    else
        id = preId
    end
    # scaling the data
    dataTX = vcat(dataUX[:, id], dataVX[:, id])
    dataTY = vcat(dataUY[:, id], dataVY[:, id])

    display(UnicodePlots.histogram(long_amps[id] * (mx_proj / mx_long), nbins=30, height=30))

    if abs(dataScale) <= sqrt(eps(typeof(dataScale)))
        dataScale_loc = sqrt(maximum(sum(dataTX .^ 2, dims=1)))
    else
        dataScale_loc = dataScale
    end
    invDS = one(dataScale_loc) ./ dataScale_loc
    println("QPPreProcess: Ratio of data returned ", length(id) / size(dataX, 2), " ", maxAmp)
    return thetaX[:, id], dataTX .* invDS, thetaY[:, id], dataTY .* invDS, dataScale_loc, id, R1, S1, W1
end

@inline function bump(x::Number, y::Number, d, c, s, f, p)
    if (-d^2 - s * x + y) < 0
        exp(y / (c * (-d^2 - s * x + y))) * (1 + 1 / (f^2 + x^p + y^p))
    else
        return zero(x)
    end
end

@inline function D1_bump(x::Number, y::Number, d, c, s, f, p)
    if (-d^2 - s * x + y) < 0
        return (-((p * x^(-1 + p)) / (f^2 + x^p + y^p)^2) + (s * y * (1 + 1 / (f^2 + x^p + y^p))) / (c * (d^2 + s * x - y)^2)) / exp(y / (c * d^2 + c * s * x - c * y))
    else
        return zero(x)
    end
end

@inline function D2_bump(x::Number, y::Number, d, c, s, f, p)
    if (-d^2 - s * x + y) < 0
        return -((p * y^(-1 + p) + ((d^2 + s * x) * (f^2 + x^p + y^p) * (1 + f^2 + x^p + y^p)) / (c * (d^2 + s * x - y)^2)) / (exp(y / (c * d^2 + c * s * x - c * y)) * (f^2 + x^p + y^p)^2))
    else
        return zero(x)
    end
end

@inline function D1D1_bump(x::Number, y::Number, d, c, s, f, p)
    if (-d^2 - s * x + y) < 0
        return ((p * x^(-2 + p) * (-(f^2 * (-1 + p)) + (1 + p) * x^p - (-1 + p) * y^p)) / (f^2 + x^p + y^p)^3 + (s * y * (s * y * (1 + 1 / (f^2 + x^p + y^p)) + 2 * c * (d^2 + s * x - y) * (-s - (p * x^(-1 + p) * (d^2 + s * x - y)) / (f^2 + x^p + y^p)^2 - s / (f^2 + x^p + y^p)))) / (c^2 * (d^2 + s * x - y)^4)) / exp(y / (c * d^2 + c * s * x - c * y))
    else
        return zero(x)
    end
end

@inline function D1D2_bump(x::Number, y::Number, d, c, s, f, p)
    if (-d^2 - s * x + y) < 0
        return (2 * c^2 * p^2 * x^(-1 + p) * (d^2 + s * x - y)^4 * y^(-1 + p) + c * p * x^(-1 + p) * (d^2 + s * x) * (d^2 + s * x - y)^2 * (f^2 + x^p + y^p) - c * p * s * (d^2 + s * x - y)^2 * y^p * (f^2 + x^p + y^p) + c * s * (d^2 + s * x - y)^2 * (f^2 + x^p + y^p)^2 * (1 + f^2 + x^p + y^p) - s * (d^2 + s * x) * y * (f^2 + x^p + y^p)^2 * (1 + f^2 + x^p + y^p) + 2 * c * s * (d^2 + s * x - y) * y * (f^2 + x^p + y^p)^2 * (1 + f^2 + x^p + y^p)) / (c^2 * exp(y / (c * d^2 + c * s * x - c * y)) * (d^2 + s * x - y)^4 * (f^2 + x^p + y^p)^3)
    else
        return zero(x)
    end
end

@inline function D2D2_bump(x::Number, y::Number, d, c, s, f, p)
    if (-d^2 - s * x + y) < 0
        return (2 * p^2 * y^(-2 + 2 * p) - (-1 + p) * p * y^(-2 + p) * (f^2 + x^p + y^p) + (2 * p * (d^2 + s * x) * y^(-1 + p) * (f^2 + x^p + y^p)) / (c * (d^2 + s * x - y)^2) + ((d^2 + s * x)^2 * (f^2 + x^p + y^p)^2 * (1 + f^2 + x^p + y^p)) / (c^2 * (d^2 + s * x - y)^4) - (2 * (d^2 + s * x) * (f^2 + x^p + y^p)^2 * (1 + f^2 + x^p + y^p)) / (c * (d^2 + s * x - y)^3)) / (exp(y / (c * d^2 + c * s * x - c * y)) * (f^2 + x^p + y^p)^3)
    else
        return zero(x)
    end
end

# ------- U -------

function QPScaleU(UoX, VoX)
    @staticscale return ones(1, size(UoX,2))
    @dynscale    return bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpU_D, bumpU_C, bumpU_S, bumpU_F, bumpU_P)
end

function D1_QPScaleU(UoX, VoX)
    @staticscale return zeros(1, size(UoX,2))
    @dynscale    return D1_bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpU_D, bumpU_C, bumpU_S, bumpU_F, bumpU_P)
end

function D2_QPScaleU(UoX, VoX)
    @staticscale return zeros(1, size(UoX,2))
    @dynscale    return D2_bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpU_D, bumpU_C, bumpU_S, bumpU_F, bumpU_P)
end

function D1D1_QPScaleU(UoX, VoX)
    @staticscale return zeros(1, size(UoX,2))
    @dynscale    return D1D1_bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpU_D, bumpU_C, bumpU_S, bumpU_F, bumpU_P)
end

function D2D2_QPScaleU(UoX, VoX)
    @staticscale return zeros(1, size(UoX,2))
    @dynscale    return D2D2_bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpU_D, bumpU_C, bumpU_S, bumpU_F, bumpU_P)
end

# ------- V -------

function QPScaleV(UoX, VoX)
    @staticscale return ones(1, size(UoX,2))
    @dynscale    return bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpV_D, bumpV_C, bumpV_S, bumpV_F, bumpV_P)
end

function D1_QPScaleV(UoX, VoX)
    @staticscale return zeros(1, size(UoX,2))
    @dynscale    return D1_bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpV_D, bumpV_C, bumpV_S, bumpV_F, bumpV_P)
end

function D2_QPScaleV(UoX, VoX)
    @staticscale return zeros(1, size(UoX,2))
    @dynscale    return D2_bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpV_D, bumpV_C, bumpV_S, bumpV_F, bumpV_P)
end

function D1D1_QPScaleV(UoX, VoX)
    @staticscale return zeros(1, size(UoX,2))
    @dynscale    return D1D1_bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpV_D, bumpV_C, bumpV_S, bumpV_F, bumpV_P)
end

function D2D2_QPScaleV(UoX, VoX)
    @staticscale return zeros(1, size(UoX,2))
    @dynscale    return D2D2_bump.(sum(UoX .^ 2, dims=1), sum(VoX .^ 2, dims=1), bumpV_D, bumpV_C, bumpV_S, bumpV_F, bumpV_P)
end

# ------- full scaling -------

function QPScaleF(MK, XK, thetaX, dataTX)
    res = dataTX .- Eval(MK, XK, thetaX)
    scale = 1 .+ 1 ./ (scale_epsilon ^ 2 .+ sum(res .^ 2, dims=1))
    return scale
end


# # ------- NORM ------- L1 - L2 hybrid
# 
# function psiNorm(x::Number, beta = norm_beta)
#     if abs(x) < beta
#         return x*x / 2 / beta
#     else
#         return abs(x) - beta/2
#     end
# end
# 
# function D_psiNorm(x::Number, beta = norm_beta)
#     if abs(x) < beta
#         return x / beta
#     else
#         return sign(x)
#     end
# end
# 
# function DD_psiNorm(x::Number, beta = norm_beta)
#     if abs(x) < beta
#         return one(x) / beta
#     else
#         return zero(x)
#     end
# end

# ------- NORM ------- L2
function psiNorm(x::Number, beta=norm_beta)
    return x * x / 2
end

function D_psiNorm(x::Number, beta=norm_beta)
    return x
end

function DD_psiNorm(x::Number, beta=norm_beta)
    return one(x)
end
# ------- CACHE -------

struct UCache
    U0
    U1
    Unl
end

struct VCache
    V0
    V1
    Vnl
end

struct UDataCache
    X::UCache
    Y::UCache
    UoX
    UoY
end

struct VDataCache
    X::VCache
    Y::VCache
    VoX
    VoY
end

struct FCache
    R
    U::UDataCache
    S
    V::VDataCache
    LU
    LV
    pU             # updates every time
    pV             # updates every time
    pF
    loss
    # only for the Hession_parts
    DF                 # the gradient
    UcoreXX            #
    UcoreXY            #
    UcoreYY            #
    VcoreXX            #
    VcoreXY            #
    VcoreYY            #
    Uvalid::Vector{Bool} # it is a single element array. If false [JS, L0_HS, G, UoX_DUoX] are invalid, the rest is updated with cache updates
    Vvalid::Vector{Bool} # it is a single element array. If false [JS, L0_HS, G, UoX_DUoX] are invalid, the rest is updated with cache updates
    Gvalid::ArrayPartition
end


@doc raw"""
    makeCache(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY)
    
Creates a cache object for a combined foliation. The cache stores intermediate results that are not re-calculated between various steps of the optimisation. This improves performance significantly.

Input:
* `M`, `X` the combined foliation generated by [`QPCombinedFoliation`](@ref)
* `dataTX` is matrix ``\boldsymbol{X}``. Each column corresponds to a data point.
* `dataTY` is matrix ``\boldsymbol{Y}``.
* `thetaX` is matrix ``\boldsymbol{\Theta}_x`` These are the interpolations vectors for Fourier collocation corresponding to ``\theta_k``.
* `thetaY` is matrix ``\boldsymbol{\Theta}_y``

The output is a cache object `cache`.
"""
function makeCache(M::QPCombinedFoliation{dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order,ℝ}, X, thetaX, dataTX, thetaY, dataTY) where {dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order}
    #     println("U0")
    X_U0 = makeCache(QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX)
    Y_U0 = makeCache(QPC_U0_manifold(M), QPC_U0_point(X), thetaY, dataTY)
    #     println("U1")
    X_U1 = makeCache(QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX)
    Y_U1 = makeCache(QPC_U1_manifold(M), QPC_U1_point(X), thetaY, dataTY)
    #     println("Unl")
    X_Unl = makeCache(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX)
    Y_Unl = makeCache(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY)
    X_U = UCache(X_U0, X_U1, X_Unl)
    Y_U = UCache(Y_U0, Y_U1, Y_Unl)
    UoX = EvalU(M, X, thetaX, dataTX; cache=X_U)
    UoY = EvalU(M, X, thetaY, dataTY; cache=Y_U)
    U = UDataCache(X_U, Y_U, UoX, UoY)
    #     println("U0")
    X_V0 = makeCache(QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX)
    Y_V0 = makeCache(QPC_V0_manifold(M), QPC_V0_point(X), thetaY, dataTY)
    #     println("U1")
    X_V1 = makeCache(QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX)
    Y_V1 = makeCache(QPC_V1_manifold(M), QPC_V1_point(X), thetaY, dataTY)
    #     println("Unl")
    X_Vnl = makeCache(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX)
    Y_Vnl = makeCache(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY)
    X_V = VCache(X_V0, X_V1, X_Vnl)
    Y_V = VCache(Y_V0, Y_V1, Y_Vnl)
    VoX = EvalV(M, X, thetaX, dataTX; cache=X_V)
    VoY = EvalV(M, X, thetaY, dataTY; cache=Y_V)
    V = VDataCache(X_V, Y_V, VoX, VoY)
    #     println("S")
    R = makeCache(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX)
    S = makeCache(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX)
    # L
    RoUoX = Eval(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX, cache=R)
    LU = RoUoX - UoY
    SoVoX = Eval(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX, cache=S)
    LV = SoVoX - VoY
    # scalings
    pF = QPScaleF(M.MK, M.XK, thetaX, dataTX)
    pU = QPScaleU(UoX, VoX) .* pF
    pV = QPScaleV(UoX, VoX) .* pF
    sU = sum(psiNorm.(LU), dims=1)
    sV = sum(psiNorm.(LV), dims=1)
    loss = sum(sU .* pU) + sum(sV .* pV)

    # these are not updated
    DF = zero(M)
    UcoreXX = zeros(dim_latent, dim_latent, size(dataTX, 2))
    UcoreXY = similar(UcoreXX)
    UcoreYY = similar(UcoreXX)
    VcoreXX = zeros(dim_data - dim_latent, dim_data - dim_latent, size(dataTX, 2))
    VcoreXY = similar(VcoreXX)
    VcoreYY = similar(VcoreXX)
    Uvalid = [false] # nothing after SEL is valid
    Vvalid = [false] # nothing after SEL is valid
    return FCache(R, U, S, V, LU, LV, pU, pV, pF, [loss], DF, UcoreXX, UcoreXY, UcoreYY, VcoreXX, VcoreXY, VcoreYY, Uvalid, Vvalid, to_bool(X))
end

@doc raw"""
    updateCache!(cache::FCache, M, X, thetaX, dataTX, thetaY, dataTY)

The same argument as [`makeCache`](@ref), expect that it updates `cache`.
"""
function updateCache!(cache::FCache, M, X, thetaX, dataTX, thetaY, dataTY)
    # S, U0, U1, Unl, B, V0 V1, Vnl, LU, LV, pU, pV, loss
    # intermediates, UoX, UoY, VoX, VoY
    @profile tms = [time()]
    updateCache!(cache.U.X.U0, QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX)
    updateCache!(cache.U.Y.U0, QPC_U0_manifold(M), QPC_U0_point(X), thetaY, dataTY)
    @profile push!(tms, time())
    updateCache!(cache.U.X.U1, QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX)
    updateCache!(cache.U.Y.U1, QPC_U1_manifold(M), QPC_U1_point(X), thetaY, dataTY)
    @profile push!(tms, time())
    updateCache!(cache.U.X.Unl, QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX)
    updateCache!(cache.U.Y.Unl, QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY)
    @profile push!(tms, time())
    cache.U.UoX .= EvalU(M, X, thetaX, dataTX; cache=cache.U.X)
    cache.U.UoY .= EvalU(M, X, thetaY, dataTY; cache=cache.U.Y)
    @profile push!(tms, time())
    updateCache!(cache.V.X.V0, QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX)
    updateCache!(cache.V.Y.V0, QPC_V0_manifold(M), QPC_V0_point(X), thetaY, dataTY)
    @profile push!(tms, time())
    updateCache!(cache.V.X.V1, QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX)
    updateCache!(cache.V.Y.V1, QPC_V1_manifold(M), QPC_V1_point(X), thetaY, dataTY)
    @profile push!(tms, time())
    updateCache!(cache.V.X.Vnl, QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX)
    updateCache!(cache.V.Y.Vnl, QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY)
    @profile push!(tms, time())
    cache.V.VoX .= EvalV(M, X, thetaX, dataTX; cache=cache.V.X)
    cache.V.VoY .= EvalV(M, X, thetaY, dataTY; cache=cache.V.Y)
    @profile push!(tms, time())
    updateCache!(cache.R, QPC_R_manifold(M), QPC_R_point(X), thetaX, cache.U.UoX)
    updateCache!(cache.S, QPC_S_manifold(M), QPC_S_point(X), thetaX, cache.V.VoX)
    @profile push!(tms, time())
    RoUoX = Eval(QPC_R_manifold(M), QPC_R_point(X), thetaX, cache.U.UoX, cache=cache.R)
    SoVoX = Eval(QPC_S_manifold(M), QPC_S_point(X), thetaX, cache.V.VoX, cache=cache.S)
    @profile push!(tms, time())
    cache.LU .= RoUoX - cache.U.UoY
    cache.LV .= SoVoX - cache.V.VoY
    @profile push!(tms, time())
    cache.pF .= QPScaleF(M.MK, M.XK, thetaX, dataTX)
    cache.pU .= QPScaleU(cache.U.UoX, cache.V.VoX) .* cache.pF
    cache.pV .= QPScaleV(cache.U.UoX, cache.V.VoX) .* cache.pF
    sU = sum(psiNorm.(cache.LU), dims=1)
    sV = sum(psiNorm.(cache.LV), dims=1)
    cache.loss .= sum(sU .* cache.pU) + sum(sV .* cache.pV)
    cache.Uvalid .= false
    cache.Vvalid .= false

    @profile push!(tms, time())
    @profile println("updateCache! times = ", tms[2:end] - tms[1:end-1])
    return nothing
end

@doc raw"""
    updateCache!(cache::FCache, M, X, thetaX, dataTX, thetaY, dataTY, sel)

The same input argument as [`makeCache`](@ref), expect that it updates `cache` only for the component of the foliation given by `sel`.
"""
function updateCache!(cache::FCache, M, X, thetaX, dataTX, thetaY, dataTY, sel)
    @profile tms = [time()]
    if sel[1] == part_R
        updateCache!(cache.R, QPC_R_manifold(M), QPC_R_point(X), thetaX, cache.U.UoX)
        # to update LU, loss
        RoUoX = Eval(QPC_R_manifold(M), QPC_R_point(X), thetaX, cache.U.UoX, cache=cache.R)
        cache.LU .= RoUoX - cache.U.UoY
        @profile push!(tms, time())
    elseif sel[1] in (part_U0, part_U1, part_Unl)
        if sel[1] == part_U0
            updateCache!(cache.U.X.U0, QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX)
            updateCache!(cache.U.Y.U0, QPC_U0_manifold(M), QPC_U0_point(X), thetaY, dataTY)
            @profile push!(tms, time())
        elseif sel[1] == part_U1
            updateCache!(cache.U.X.U1, QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX)
            updateCache!(cache.U.Y.U1, QPC_U1_manifold(M), QPC_U1_point(X), thetaY, dataTY)
            @profile push!(tms, time())
        elseif sel[1] == part_Unl
            if QPC_Unl_manifold(M) isa QPPolynomial
                updateCache!(cache.U.X.Unl, QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX)
                updateCache!(cache.U.Y.Unl, QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY)
            else
                updateCache!(cache.U.X.Unl, QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX, sel[2:end])
                updateCache!(cache.U.Y.Unl, QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY, sel[2:end])
            end
            @profile push!(tms, time())
        end
        # to update UoX, UoY, VoX, VoY, S, Vnl, B, LU, LV, pU, pV, loss : everything except (two of U0, U1, Unl), V0, V1
        cache.U.UoX .= EvalU(M, X, thetaX, dataTX, cache=cache.U.X)
        cache.U.UoY .= EvalU(M, X, thetaY, dataTY, cache=cache.U.Y)
        updateCache!(cache.R, QPC_R_manifold(M), QPC_R_point(X), thetaX, cache.U.UoX)
        RoUoX = Eval(QPC_R_manifold(M), QPC_R_point(X), thetaX, cache.U.UoX, cache=cache.R)
        cache.LU .= RoUoX - cache.U.UoY
        cache.pU .= QPScaleU(cache.U.UoX, cache.V.VoX) .* cache.pF
        cache.pV .= QPScaleV(cache.U.UoX, cache.V.VoX) .* cache.pF
        @profile push!(tms, time())
    elseif sel[1] == part_S
        updateCache!(cache.S, QPC_S_manifold(M), QPC_S_point(X), thetaX, cache.V.VoX)
        # to update LV, loss
        SoVoX = Eval(QPC_S_manifold(M), QPC_S_point(X), thetaX, cache.V.VoX, cache=cache.S)
        cache.LV .= SoVoX - cache.V.VoY
        @profile push!(tms, time())
    elseif sel[1] in (part_V0, part_V1, part_Vnl)
        if sel[1] == part_V0
            updateCache!(cache.V.X.V0, QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX)
            updateCache!(cache.V.Y.V0, QPC_V0_manifold(M), QPC_V0_point(X), thetaY, dataTY)
            @profile push!(tms, time())
        elseif sel[1] == part_V1
            updateCache!(cache.V.X.V1, QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX)
            updateCache!(cache.V.Y.V1, QPC_V1_manifold(M), QPC_V1_point(X), thetaY, dataTY)
            @profile push!(tms, time())
        elseif sel[1] == part_Vnl
            if QPC_Vnl_manifold(M) isa QPPolynomial
                updateCache!(cache.V.X.Vnl, QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX)
                updateCache!(cache.V.Y.Vnl, QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY)
            else
                updateCache!(cache.V.X.Vnl, QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX, sel[2:end])
                updateCache!(cache.V.Y.Vnl, QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY, sel[2:end])
            end
            @profile push!(tms, time())
        end
        # to update VoX, VoY, B, LV, pU, pV, loss
        cache.V.VoX .= EvalV(M, X, thetaX, dataTX; cache=cache.V.X)
        cache.V.VoY .= EvalV(M, X, thetaY, dataTY; cache=cache.V.Y)
        updateCache!(cache.S, QPC_S_manifold(M), QPC_S_point(X), thetaX, cache.V.VoX)
        SoVoX = Eval(QPC_S_manifold(M), QPC_S_point(X), thetaX, cache.V.VoX, cache=cache.S)
        cache.LV .= SoVoX - cache.V.VoY
        cache.pU .= QPScaleU(cache.U.UoX, cache.V.VoX) .* cache.pF
        cache.pV .= QPScaleV(cache.U.UoX, cache.V.VoX) .* cache.pF
        @profile push!(tms, time())
    else
        println("wrong sel")
    end
    sU = sum(psiNorm.(cache.LU), dims=1)
    sV = sum(psiNorm.(cache.LV), dims=1)
    cache.loss .= sum(sU .* cache.pU) + sum(sV .* cache.pV)
    cache.Uvalid .= false
    cache.Vvalid .= false
    @profile push!(tms, time())
    @profile println("updateCache!(sel) times = ", tms[2:end] - tms[1:end-1])
    return nothing
end

@doc raw"""
    EvalU(M::QPCombinedFoliation, X, thetaX, dataTX; cache)

Evaluates the submersion ``\boldsymbol{U}`` at data points `thetaX`, `dataTX`.
"""
function EvalU(M::QPCombinedFoliation, X, thetaX, dataTX;
    cache=UCache(makeCache(QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX),
        makeCache(QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX),
        makeCache(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX)))
    UoX = (Eval(QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX, cache=cache.U0)
           + Eval(QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX, cache=cache.U1)
           + Eval(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX, cache=cache.Unl))
    return UoX
end

@doc raw"""
    EvalV(M::QPCombinedFoliation, X, thetaX, dataTX; cache)

Evaluates the submersion ``\boldsymbol{V}`` at data points `thetaX`, `dataTX`.
"""
function EvalV(M::QPCombinedFoliation, X, thetaX, dataTX;
    cache=VCache(makeCache(QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX),
        makeCache(QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX),
        makeCache(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX)))
    VoX = (Eval(QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX, cache=cache.V0)
           + Eval(QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX, cache=cache.V1)
           + Eval(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX, cache=cache.Vnl))
    return VoX
end

# TODO: needs rework
# this is to be used when the projection is not available
# so the projection needs to be performed before the evaluation
function EvalUnl(M::QPCombinedFoliation{dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order,ℝ}, X, t::Number, data::Vector) where {dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order}
    theta = InterpolationWeights(QPC_Unl_manifold(M), t)
    return vec(Eval(QPC_Unl_manifold(M), QPC_Unl_point(X), reshape(theta, :, 1), reshape(data, :, 1)))
end

function EvalVnl(M::QPCombinedFoliation{dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order,ℝ}, X, t::Number, data::Vector) where {dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order}
    theta = InterpolationWeights(QPC_Vnl_manifold(M), t)
    return vec(Eval(QPC_Vnl_manifold(M), QPC_Vnl_point(X), reshape(theta, :, 1), reshape(data, :, 1)))
end

@doc raw"""
    ResidualU(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY; cache)

Returns the residual of the invariance equation for ``\boldsymbol{U}`` divided by the norm of ``\boldsymbol{V}``.
"""
function ResidualU(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY;
    cache=makeCache(M, X, thetaX, dataTX, thetaY, dataTY), bins = (40,20))
    res = dataTX .- Eval(M.MK, M.XK, thetaX)
    LU = cache.LU
    return vec(sqrt.(sum(LU .^ 2, dims=1) ./ sum(res .^ 2, dims=1))), vec(cache.pU), vec(cache.pV)
end

@doc raw"""
    ErrorStatistics(M::QPCombinedFoliation, X, MW::QPPolynomial, XW, thetaX, dataTX, thetaY, dataTY; dataScale=1.0, bins=(40,20), cache)
    
Calculates some statistics about the fitted foliation and the data distribution with respect to the recovered invariant manifold.

Input
* `M`, `X` is the combined foliation [`QPCombinedFoliation`](@ref)
* `MW`, `XW` is the recovered manifold immersion (see [`QPPostProcess`](@ref)
* `thetaX`, `dataTX`, `thetaY`, `dataTY` is the data
* `dataScale` the scaling factor that was used to divide the data with
* `bins` is a tuple specifying the number of bins 1. in the direction of the amplitude 2. in the direction of the error
* `cache` (optional) the data cache created by [`makeCache`](@ref)

Output
```
OnManifoldAmplitude, hsU, errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX
```
* `OnManifoldAmplitude` the amplitude of each data point as projected to the invariant manifold
* `hsU` the density of the data points at various amplitude levels
* `errMaxX`, `errMaxY` the maximum error as a function of the amplitude on the manifold. X is error Y is amplitude
* `errMinX`, `errMinY` the minimum error as a function of the amplitude on the manifold. X is error Y is amplitude
* `errMeanX`, `errMeanY` the mean error as a function of the amplitude on the manifold. X is error Y is amplitude
* `errStdX` standard deviation of the error
"""
function ErrorStatistics(MCF::QPCombinedFoliation, XCF, MW::QPPolynomial, XW, thetaTX, dataTX, thetaTY, dataTY;
                         dataScale = 1.0,
                         bins = (40,20), maxAmp=Inf,
                         cache=makeCache(M, X, thetaTX, dataTX, thetaTY, dataTY))
    
    UoX = cache.U.UoX
    OnManifoldAmplitude = sqrt.(dropdims(sum((Eval(MW, XW, thetaTX, UoX)) .^ 2, dims=1), dims=1)) .* dataScale

    # error statistics
    resU, residU, residV = ResidualU(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY)
    minU = eps(1.0)
    maxU = min(maximum(OnManifoldAmplitude), maxAmp)
    edU = (exp.(range(log(minimum(resU)), log(maximum(resU)), length=bins[1])), range(minU, maxU, length=bins[2]))
    hsU = fit(Histogram{Float64}, (resU, OnManifoldAmplitude), UnitWeights{Float64}(length(resU)), edU)
    nzinv = x -> iszero(x) ? 1 : 1 / x
    hsU.weights .*= nzinv.(sum(hsU.weights, dims=1))

    pp = collect((hsU.edges[2][1:end-1] + hsU.edges[2][2:end]) / 2)
    qq = collect((hsU.edges[1][1:end-1] + hsU.edges[1][2:end]) / 2)
    maxidx = [findlast(hsU.weights[:, k] .> 0) for k in axes(hsU.weights, 2)]
    minidx = [findfirst(hsU.weights[:, k] .> 0) for k in axes(hsU.weights, 2)]
    maxok = findall(maxidx .!= nothing)
    minok = findall(minidx .!= nothing)
    errMaxX = qq[maxidx[maxok]]
    errMaxY = pp[maxok]
    errMinX = qq[minidx[minok]]
    errMinY = pp[minok]
    errMeanX = [mean(qq, weights(hsU.weights[:, k])) for k in axes(hsU.weights, 2)]
    errMeanY = deepcopy(pp)
    errStdX = [std(qq, weights(hsU.weights[:, k])) for k in axes(hsU.weights, 2)]
    hsU.edges[1][1] = hsU.edges[1][2]
    
    return OnManifoldAmplitude, hsU, errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX
end

@doc raw"""
    ResidualLoss(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY; cache)

Returns the loss function only for the component with ``\boldsymbol{U}`` of the combined foliation `M`, `X` at data points `thetaX`, `dataTX`, `thetaY`, `dataTY`.
"""
function ResidualLoss(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY;
    cache=makeCache(M, X, thetaX, dataTX, thetaY, dataTY))
    LU = cache.LU
    return vec(sum(psiNorm.(LU), dims=1) .* cache.pU), vec(cache.pU), vec(cache.pV)
end

@doc raw"""
    Loss(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY; cache)

Returns the loss function of the combined foliation `M`, `X` at data points `thetaX`, `dataTX`, `thetaY`, `dataTY`.
"""
function Loss(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY;
    cache=makeCache(M, X, thetaX, dataTX, thetaY, dataTY))
    return cache.loss[1]
end

# copied over from below and modified...
function Gradient(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY; riemannian=true,
    cache=makeCache(M, X, thetaX, dataTX, thetaY, dataTY))
    UoX = cache.U.UoX
    UoY = cache.U.UoY
    LU = cache.LU
    VoX = cache.V.VoX
    VoY = cache.V.VoY
    LV = cache.LV
    pF = cache.pF
    pU = cache.pU
    pV = cache.pV
    DsU = D_psiNorm.(LU)
    DsV = D_psiNorm.(LV)

    cache.Gvalid .= true
    let sel = (part_R,)
        DR = L0_DF(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX, DsU .* pU, cache=cache.R)
        # no need for projection, it is Euclidean
        Set_parts!(cache.DF, DR, sel)
    end
    let sel = (part_S,)
        DS = L0_DF(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX, DsV .* pV, cache=cache.S)
        # no need for projection, it is Euclidean
        Set_parts!(cache.DF, DS, sel)
    end
    begin
        @dynscale D1pU = D1_QPScaleU(UoX, VoX) .* pF
        @dynscale D1pV = D1_QPScaleV(UoX, VoX) .* pF
        @dynscale sU = sum(psiNorm.(LU), dims=1)
        @dynscale sV = sum(psiNorm.(LV), dims=1)
        GoX = L0_JF(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX, DsU .* pU, cache=cache.R)
        @dynscale GoX .+= 2 .* (D1pU .* sU .+ D1pV .* sV) .* UoX
        GoY = pU .* DsU
        # parts...
        let sel = (part_U0,)
            DU = L0_DF(QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX, GoX, cache=cache.U.X.U0) # dataUX disregarded
            DU -= L0_DF(QPC_U0_manifold(M), QPC_U0_point(X), thetaY, dataTY, GoY, cache=cache.U.Y.U0)
            Set_parts!(cache.DF, DU, sel)
        end
        let sel = (part_U1,)
            DU = L0_DF(QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX, GoX, cache=cache.U.X.U1)
            DU -= L0_DF(QPC_U1_manifold(M), QPC_U1_point(X), thetaY, dataTY, GoY, cache=cache.U.Y.U1)
            Set_parts!(cache.DF, DU, sel)
        end
        let sel = (part_Unl,)
            DU = L0_DF(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX, GoX, cache=cache.U.X.Unl)
            DU -= L0_DF(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY, GoY, cache=cache.U.Y.Unl)
            Set_parts!(cache.DF, DU, sel)
        end
    end
    begin
        @dynscale D2pU = D2_QPScaleU(UoX, VoX) .* pF
        @dynscale D2pV = D2_QPScaleV(UoX, VoX) .* pF
        @dynscale sU = sum(psiNorm.(LU), dims=1)
        @dynscale sV = sum(psiNorm.(LV), dims=1)
        GoX = L0_JF(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX, DsV .* pV, cache=cache.S)
        @dynscale GoX .+= 2 .* (D2pU .* sU .+ D2pV .* sV) .* VoX
        GoY = pV .* DsV
        let sel = (part_V0,)
            DV = L0_DF(QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX, GoX, cache=cache.V.X.V0)
            DV -= L0_DF(QPC_V0_manifold(M), QPC_V0_point(X), thetaY, dataTY, GoY, cache=cache.V.Y.V0)
            Set_parts!(cache.DF, DV, sel)
        end
        let sel = (part_V1,)
            DV = L0_DF(QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX, GoX, cache=cache.V.X.V1)
            DV -= L0_DF(QPC_V1_manifold(M), QPC_V1_point(X), thetaY, dataTY, GoY, cache=cache.V.Y.V1)
            Set_parts!(cache.DF, DV, sel)
        end
        let sel = (part_Vnl,)
            DV = L0_DF(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX, GoX, cache=cache.V.X.Vnl)
            DV -= L0_DF(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY, GoY, cache=cache.V.Y.Vnl)
            Set_parts!(cache.DF, DV, sel)
        end
    end
    if riemannian
        return project(M, X, cache.DF)
    else
        return cache.DF
    end
end

# if sel is empty, it should fill up the cache with the full gradient
# riemannian = false => do not perform projection to the tangent space. Helpful when testing accuracy
function Gradient(M::QPCombinedFoliation, X, thetaX, dataTX, thetaY, dataTY, sel; riemannian=true,
    cache=makeCache(M, X, thetaX, dataTX, thetaY, dataTY))
    UoX = cache.U.UoX
    UoY = cache.U.UoY
    LU = cache.LU
    VoX = cache.V.VoX
    VoY = cache.V.VoY
    LV = cache.LV
    pF = cache.pF
    pU = cache.pU
    pV = cache.pV
    DsU = D_psiNorm.(LU)
    DsV = D_psiNorm.(LV)

    Set_parts!(cache.Gvalid, true, sel)
    if sel[1] == part_R
        DR = L0_DF(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX, DsU .* pU, cache=cache.R)
        # no need for projection, it is Euclidean
        Set_parts!(cache.DF, DR, sel)
        return DR
    elseif sel[1] == part_S
        DS = L0_DF(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX, DsV .* pV, cache=cache.S)
        # no need for projection, it is Euclidean
        Set_parts!(cache.DF, DS, sel)
        return DS
    elseif sel[1] in (part_U0, part_U1, part_Unl)
        @dynscale D1pU = D1_QPScaleU(UoX, VoX) .* pF
        @dynscale D1pV = D1_QPScaleV(UoX, VoX) .* pF
        @dynscale sU = sum(psiNorm.(LU), dims=1)
        @dynscale sV = sum(psiNorm.(LV), dims=1)
        GoX = L0_JF(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX, DsU .* pU, cache=cache.R)
        @dynscale GoX .+= 2 .* (D1pU .* sU .+ D1pV .* sV) .* UoX
        GoY = pU .* DsU
        # parts...
        if sel[1] == part_U0
            DU = L0_DF(QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX, GoX, cache=cache.U.X.U0) # dataUX disregarded
            DU -= L0_DF(QPC_U0_manifold(M), QPC_U0_point(X), thetaY, dataTY, GoY, cache=cache.U.Y.U0)
            Set_parts!(cache.DF, DU, sel)
            return DU
        elseif sel[1] == part_U1
            DU = L0_DF(QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX, GoX, cache=cache.U.X.U1)
            DU -= L0_DF(QPC_U1_manifold(M), QPC_U1_point(X), thetaY, dataTY, GoY, cache=cache.U.Y.U1)
            Set_parts!(cache.DF, DU, sel)
            if riemannian
                return project(QPC_U1_manifold(M), QPC_U1_point(X), DU)
            else
                return DU
            end
        elseif sel[1] == part_Unl
            if QPC_Unl_manifold(M) isa QPPolynomial
                DU = L0_DF(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX, GoX, cache=cache.U.X.Unl)
                DU -= L0_DF(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY, GoY, cache=cache.U.Y.Unl)
                Set_parts!(cache.DF, DU, sel)
                return DU # no projection
            else
                DU = L0_DF_parts(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX, GoX, sel[2:end], cache=cache.U.X.Unl)
                DU -= L0_DF_parts(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY, GoY, sel[2:end], cache=cache.U.Y.Unl)
                Set_parts!(cache.DF, DU, sel)
                if riemannian
                    return project(QPC_Unl_manifold(M).M.manifolds[sel[2]].M.manifolds[sel[3]], QPC_Unl_point(X).x[sel[2]].x[sel[3]], DU)
                else
                    return DU
                end
            end
        end
    elseif sel[1] in (part_V0, part_V1, part_Vnl)
        @dynscale D2pU = D2_QPScaleU(UoX, VoX) .* pF
        @dynscale D2pV = D2_QPScaleV(UoX, VoX) .* pF
        @dynscale sU = sum(psiNorm.(LU), dims=1)
        @dynscale sV = sum(psiNorm.(LV), dims=1)
        GoX = L0_JF(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX, DsV .* pV, cache=cache.S)
        @dynscale GoX .+= 2 .* (D2pU .* sU .+ D2pV .* sV) .* VoX
        GoY = pV .* DsV
        if sel[1] == part_V0
            DV = L0_DF(QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX, GoX, cache=cache.V.X.V0)
            DV -= L0_DF(QPC_V0_manifold(M), QPC_V0_point(X), thetaY, dataTY, GoY, cache=cache.V.Y.V0)
            Set_parts!(cache.DF, DV, sel)
            return DV
        elseif sel[1] == part_V1
            DV = L0_DF(QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX, GoX, cache=cache.V.X.V1)
            DV -= L0_DF(QPC_V1_manifold(M), QPC_V1_point(X), thetaY, dataTY, GoY, cache=cache.V.Y.V1)
            Set_parts!(cache.DF, DV, sel)
            if riemannian
                return project(QPC_V1_manifold(M), QPC_V1_point(X), DV)
            else
                return DV
            end
        elseif sel[1] == part_Vnl
            if QPC_Vnl_manifold(M) isa QPPolynomial
                DV = L0_DF(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX, GoX, cache=cache.V.X.Vnl)
                DV -= L0_DF(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY, GoY, cache=cache.V.Y.Vnl)
                Set_parts!(cache.DF, DV, sel)
                return DV
            else
                DV = L0_DF_parts(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX, GoX, sel[2:end], cache=cache.V.X.Vnl)
                DV -= L0_DF_parts(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY, GoY, sel[2:end], cache=cache.V.Y.Vnl)
                Set_parts!(cache.DF, DV, sel)
                if riemannian
                    return project(QPC_Vnl_manifold(M).M.manifolds[sel[2]].M.manifolds[sel[3]], QPC_Vnl_point(X).x[sel[2]].x[sel[3]], DV)
                else
                    return DV
                end
            end
        end
    end
end

function HessDX(coreXX, coreXY, coreYY, DUoX_dx, DUoY_dx)
    @tullio pUX[i, k] := coreXX[i, j, k] * DUoX_dx[j, k]
    @tullio pUX[i, k] += coreXY[i, j, k] * DUoY_dx[j, k]
    # DUy
    @tullio pUY[i, k] := coreXY[j, i, k] * DUoX_dx[j, k]
    @tullio pUY[i, k] += coreYY[i, j, k] * DUoY_dx[j, k]
    return pUX, pUY
end

function Hessian(M::QPCombinedFoliation, X, dx, thetaX, dataTX, thetaY, dataTY, sel; riemannian=true,
    cache=makeCache(M, X, thetaX, dataTX, thetaY, dataTY))
    UoX = cache.U.UoX
    UoY = cache.U.UoY
    LU = cache.LU
    VoX = cache.V.VoX
    VoY = cache.V.VoY
    LV = cache.LV
    pF = cache.pF
    pU = cache.pU
    pV = cache.pV
    @dynscale sU = sum(psiNorm.(LU), dims=1)
    @dynscale sV = sum(psiNorm.(LV), dims=1)
    DsU = D_psiNorm.(LU)
    DsV = D_psiNorm.(LV)

    if sel[1] == part_R
        DDsU = DD_psiNorm.(LU)
        DR_dx = DF_dt(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX; dt=dx, cache=cache.R)
        DDR = L0_DF(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX, DDsU .* DR_dx .* pU, cache=cache.R)
        return DDR
        # note if not Euclidean, this depends on the gradient, too!
    elseif sel[1] == part_S
        DDsV = DD_psiNorm.(LV)
        DS_dx = DF_dt(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX; dt=dx, cache=cache.S)
        DDS = L0_DF(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX, DDsV .* DS_dx .* pV, cache=cache.S)
        return DDS
    elseif sel[1] in (part_U0, part_U1, part_Unl)
        @profile tms = [time()]
        if !cache.Uvalid[1]
            IU = Diagonal(I, size(UoX, 1))

            @dynscale D1pU = D1_QPScaleU(UoX, VoX) .* pF
            @dynscale D1D1pU = D1D1_QPScaleU(UoX, VoX) .* pF
            @dynscale D1pV = D1_QPScaleV(UoX, VoX) .* pF
            @dynscale D1D1pV = D1D1_QPScaleV(UoX, VoX) .* pF

            DDsU = DD_psiNorm.(LU)

            JR = JF(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX, cache=cache.R)
            pU_DsU_HR = L0_HF(QPC_R_manifold(M), QPC_R_point(X), thetaX, UoX, DsU .* pU, cache=cache.R)
            #------------------------------------------------
            # D^2_U L
            #------------------------------------------------
            @tullio           XXS[i, j, k] := pU_DsU_HR[i, j, k]                                   # S3
            @tullio           XXS[i, j, k] += JR[p, i, k] * DDsU[p, k] * JR[p, j, k] * pU[k]       # S4
            @dynscale @tullio XXS[i, j, k] += 2 * IU[i, j] * D1pU[k] * sU[k]                       # S1
            @dynscale @tullio XXS[i, j, k] += 4 * UoX[i, k] * UoX[j, k] * D1D1pU[k] * sU[k]        # S2
            @dynscale @tullio XXS[i, j, k] += 2 * IU[i, j] * D1pV[k] * sV[k]                       # S5
            @dynscale @tullio XXS[i, j, k] += 4 * UoX[i, k] * UoX[j, k] * D1D1pV[k] * sV[k]        # S6
            @dynscale @tullio XXA[i, j, k] := 2 * UoX[i, k] * (DsU[p, k] * JR[p, j, k]) * D1pU[k]  # A1-1
            @tullio           XY[i, j, k] := -JR[j, i, k] * DDsU[j, k] * pU[k]                     # XY2
            @dynscale @tullio XY[i, j, k] += -2 * UoX[i, k] * DsU[j, k] * D1pU[k]                  # XY1
            @tullio           YY[i, j, k] := DDsU[i, k] * IU[i, j] * pU[k]                         #
            # cores
            cache.UcoreXX .= XXS
            @dynscale cache.UcoreXX .+= XXA .+ permutedims(XXA, (2, 1, 3))
            cache.UcoreXY .= XY
            cache.UcoreYY .= YY
            cache.Uvalid .= true
        end
        @profile push!(tms, time())
        # the remaining bits
        if sel[1] == part_U0
            DUoX_dx = DF_dt(QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX; dt=dx, cache=cache.U.X.U0)  # (1) dataUX is ignored
            DUoY_dx = DF_dt(QPC_U0_manifold(M), QPC_U0_point(X), thetaY, dataTY; dt=dx, cache=cache.U.Y.U0)  # (2)
            pUX, pUY = HessDX(cache.UcoreXX, cache.UcoreXY, cache.UcoreYY, DUoX_dx, DUoY_dx)
            HX = L0_DF(QPC_U0_manifold(M), QPC_U0_point(X), thetaX, dataTX, pUX, cache=cache.U.X.U0)
            HY = L0_DF(QPC_U0_manifold(M), QPC_U0_point(X), thetaY, dataTY, pUY, cache=cache.U.Y.U0)
            @profile push!(tms, time())
            #             @profile println("U HESS times = ", tms[2:end] - tms[1:end-1])
            return HX + HY
        elseif sel[1] == part_U1
            DUoX_dx = DF_dt(QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX; dt=dx, cache=cache.U.X.U1)  # (1)
            DUoY_dx = DF_dt(QPC_U1_manifold(M), QPC_U1_point(X), thetaY, dataTY; dt=dx, cache=cache.U.Y.U1)  # (2)
            pUX, pUY = HessDX(cache.UcoreXX, cache.UcoreXY, cache.UcoreYY, DUoX_dx, DUoY_dx)
            HX = L0_DF(QPC_U1_manifold(M), QPC_U1_point(X), thetaX, dataTX, pUX, cache=cache.U.X.U1)
            HY = L0_DF(QPC_U1_manifold(M), QPC_U1_point(X), thetaY, dataTY, pUY, cache=cache.U.Y.U1)
            @profile push!(tms, time())
            #             @profile println("U HESS times = ", tms[2:end] - tms[1:end-1])
            if riemannian
                return HessianProjection(QPC_U1_manifold(M), QPC_U1_point(X), QPC_U1_point(cache.DF), HX + HY, dx)
            else
                return HX + HY
            end
        elseif sel[1] == part_Unl
            if QPC_Unl_manifold(M) isa QPPolynomial
                DUoX_dx = DF_dt(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX; dt=dx, cache=cache.U.X.Unl)  # (1)
                DUoY_dx = DF_dt(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY; dt=dx, cache=cache.U.Y.Unl)  # (2)
                pUX, pUY = HessDX(cache.UcoreXX, cache.UcoreXY, cache.UcoreYY, DUoX_dx, DUoY_dx)
                HX = L0_DF(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX, pUX; cache=cache.U.X.Unl)
                HY = L0_DF(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY, pUY; cache=cache.U.Y.Unl)
                @profile push!(tms, time())
                #                 @profile println("Unl HESS times = ", tms[2:end] - tms[1:end-1])
                return HX + HY
            else
                DUoX_dx = DF_dt_parts(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX, sel[2:end]; dt=dx, cache=cache.U.X.Unl)  # (1)
                DUoY_dx = DF_dt_parts(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY, sel[2:end]; dt=dx, cache=cache.U.Y.Unl)  # (2)
                pUX, pUY = HessDX(cache.UcoreXX, cache.UcoreXY, cache.UcoreYY, DUoX_dx, DUoY_dx)
                HX = L0_DF_parts(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaX, dataTX, pUX, sel[2:end]; cache=cache.U.X.Unl)
                HY = L0_DF_parts(QPC_Unl_manifold(M), QPC_Unl_point(X), thetaY, dataTY, pUY, sel[2:end]; cache=cache.U.Y.Unl)
                @profile push!(tms, time())
                @profile println("U HESS times = ", tms[2:end] - tms[1:end-1])
                if riemannian
                    return HessianProjection(QPC_Unl_manifold(M).M.manifolds[sel[2]].M.manifolds[sel[3]], QPC_Unl_point(X).x[sel[2]].x[sel[3]], Get_parts(cache.DF, sel), HX + HY, dx)
                else
                    return HX + HY
                end
            end
        end
    elseif sel[1] in (part_V0, part_V1, part_Vnl)
        tms = [time()]
        if !cache.Vvalid[1]
            IV = Diagonal(I, size(VoX, 1))

            @dynscale D2pU = D2_QPScaleU(UoX, VoX) .* pF
            @dynscale D2D2pU = D2D2_QPScaleU(UoX, VoX) .* pF
            @dynscale D2pV = D2_QPScaleV(UoX, VoX) .* pF
            @dynscale D2D2pV = D2D2_QPScaleV(UoX, VoX) .* pF

            DDsV = DD_psiNorm.(LV)

            JS = JF(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX, cache=cache.S)
            pV_DsV_HS = L0_HF(QPC_S_manifold(M), QPC_S_point(X), thetaX, VoX, DsV .* pV, cache=cache.S)
            #------------------------------------------------
            # D^2_V L
            #------------------------------------------------
            @tullio           XXS[i, j, k] := pV_DsV_HS[i, j, k]                                   # S3    OK
            @tullio           XXS[i, j, k] += JS[p, i, k] * DDsV[p, k] * JS[p, j, k] * pV[k]       # S4    OK
            @dynscale @tullio XXS[i, j, k] += 2 * IV[i, j] * D2pV[k] * sV[k]                       # S1    OK
            @dynscale @tullio XXS[i, j, k] += 4 * VoX[i, k] * VoX[j, k] * D2D2pV[k] * sV[k]        # S2
            @dynscale @tullio XXS[i, j, k] += 2 * IV[i, j] * D2pU[k] * sU[k]                       # S5
            @dynscale @tullio XXS[i, j, k] += 4 * VoX[i, k] * VoX[j, k] * D2D2pU[k] * sU[k]        # S6    OK
            @dynscale @tullio XXA[i, j, k] := 2 * VoX[i, k] * (DsV[p, k] * JS[p, j, k]) * D2pV[k]  # A1-1  OK
            @tullio           XY[i, j, k] := -JS[j, i, k] * DDsV[j, k] * pV[k]                     # XY2   OK
            @dynscale @tullio XY[i, j, k] += -2 * VoX[i, k] * DsV[j, k] * D2pV[k]                  # XY1   OK
            @tullio           YY[i, j, k] := DDsV[i, k] * IV[i, j] * pV[k]                         #       OK
            # cores
            cache.VcoreXX .= XXS
            @dynscale cache.VcoreXX .+= XXA .+ permutedims(XXA, (2, 1, 3))
            cache.VcoreXY .= XY
            cache.VcoreYY .= YY
            cache.Vvalid .= true
        end
        @profile push!(tms, time())
        if sel[1] == part_V0
            DUoX_dx = DF_dt(QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX; dt=dx, cache=cache.V.X.V0)  # (1) dataVX is ignored
            DUoY_dx = DF_dt(QPC_V0_manifold(M), QPC_V0_point(X), thetaY, dataTY; dt=dx, cache=cache.V.Y.V0)  # (2)
            pUX, pUY = HessDX(cache.VcoreXX, cache.VcoreXY, cache.VcoreYY, DUoX_dx, DUoY_dx)
            HX = L0_DF(QPC_V0_manifold(M), QPC_V0_point(X), thetaX, dataTX, pUX, cache=cache.V.X.V0)
            HY = L0_DF(QPC_V0_manifold(M), QPC_V0_point(X), thetaY, dataTY, pUY, cache=cache.V.Y.V0)
            #             @profile push!(tms, time())
            #             @profile println("V HESS times = ", tms[2:end] - tms[1:end-1])
            return HX + HY
        elseif sel[1] == part_V1
            DUoX_dx = DF_dt(QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX; dt=dx, cache=cache.V.X.V1)  # (1)
            DUoY_dx = DF_dt(QPC_V1_manifold(M), QPC_V1_point(X), thetaY, dataTY; dt=dx, cache=cache.V.Y.V1)  # (2)
            pUX, pUY = HessDX(cache.VcoreXX, cache.VcoreXY, cache.VcoreYY, DUoX_dx, DUoY_dx)
            HX = L0_DF(QPC_V1_manifold(M), QPC_V1_point(X), thetaX, dataTX, pUX, cache=cache.V.X.V1)
            HY = L0_DF(QPC_V1_manifold(M), QPC_V1_point(X), thetaY, dataTY, pUY, cache=cache.V.Y.V1)
            #             @profile push!(tms, time())
            #             @profile println("V HESS times = ", tms[2:end] - tms[1:end-1])
            if riemannian
                return HessianProjection(QPC_V1_manifold(M), QPC_V1_point(X), QPC_V1_point(cache.DF), HX + HY, dx)
            else
                return HX + HY
            end
        elseif sel[1] == part_Vnl
            if QPC_Vnl_manifold(M) isa QPPolynomial
                DUoX_dx = DF_dt(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX; dt=dx, cache=cache.V.X.Vnl)  # (1)
                @profile push!(tms, time())
                DUoY_dx = DF_dt(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY; dt=dx, cache=cache.V.Y.Vnl)  # (2)
                @profile push!(tms, time())
                pUX, pUY = HessDX(cache.VcoreXX, cache.VcoreXY, cache.VcoreYY, DUoX_dx, DUoY_dx)
                @profile push!(tms, time())
                HX = L0_DF(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX, pUX, cache=cache.V.X.Vnl)
                @profile push!(tms, time())
                HY = L0_DF(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY, pUY, cache=cache.V.Y.Vnl)
                #             @profile push!(tms, time())
                #             @profile println("V HESS times = ", tms[2:end] - tms[1:end-1])
                return HX + HY
            else
                DUoX_dx = DF_dt_parts(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX, sel[2:end]; dt=dx, cache=cache.V.X.Vnl)  # (1)
                @profile push!(tms, time())
                DUoY_dx = DF_dt_parts(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY, sel[2:end]; dt=dx, cache=cache.V.Y.Vnl)  # (2)
                @profile push!(tms, time())
                pUX, pUY = HessDX(cache.VcoreXX, cache.VcoreXY, cache.VcoreYY, DUoX_dx, DUoY_dx)
                @profile push!(tms, time())
                HX = L0_DF_parts(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaX, dataTX, pUX, sel[2:end]; cache=cache.V.X.Vnl)
                @profile push!(tms, time())
                HY = L0_DF_parts(QPC_Vnl_manifold(M), QPC_Vnl_point(X), thetaY, dataTY, pUY, sel[2:end]; cache=cache.V.Y.Vnl)
                #             @profile push!(tms, time())
                #             @profile println("V HESS times = ", tms[2:end] - tms[1:end-1])
                if riemannian
                    return HessianProjection(QPC_Vnl_manifold(M).M.manifolds[sel[2]].M.manifolds[sel[3]], QPC_Vnl_point(X).x[sel[2]].x[sel[3]], Get_parts(cache.DF, sel), HX + HY, dx)
                else
                    return HX + HY
                end
            end
        end
    else
        println("UNRECOGNISED PART")
    end
end

# this works for a single mode
function findFrequency(MF::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, XF, omega) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    grid = getgrid(fourier_order)
    B = GetLinearPart(MF, XF)
    T = transferOperator(grid, B, omega)
    eva, eve = eigen(T)
    # right bundles 1st: n - dimension 2nd: collocation point, 3rd: index of eigenvector
    eve_reshape = reshape(eve, dim_in, :, size(eve, 2))
    FM = FourierMatrix(grid)
    @tullio eve_fourier[p, l, q] := FM[l, k] * eve_reshape[p, k, q]
    eve_fourier_amplitude = sqrt.(dropdims(sum(real.(eve_fourier .* conj(eve_fourier)), dims=1), dims=1))
    # normalise for the maximum value
    eve_fourier_amplitude ./= sqrt.(sum(eve_fourier_amplitude .^ 2, dims=1))
    # finding the most central ones
    # maybe better just to find one with the highest coefficient at the lowest frequency?
    scales = [2^(abs(k)) for k in -fourier_order:fourier_order]
    roughness = transpose(eve_fourier_amplitude) * scales
    m1, id1 = findmin(roughness)
    return eva[id1]
end

mutable struct QPDebugTRO <: DebugAction
    print::Function
    t0
    Tstep
    omega
    radius
    sel
    name
    QPDebugTRO(Tstep, omega, radius, sel, print::Function=print; d_name="data") = new(print, time(), Tstep, omega, [radius], sel, d_name)
end

# we need to calculate the actual frequency to check progress at least at the linear level
function (d::QPDebugTRO)(mp, trs, i::Int)
    d.radius[1] = trs.trust_region_radius
    txt = ("        $(i). " *
           @sprintf("time = %.1f[s] ", time() - d.t0) *
           @sprintf("F(x) = %.5e ", get_cost(mp, trs.p)) *
           @sprintf("G(x) = %.5e ", norm(get_manifold(mp), trs.p, trs.X)) *
           @sprintf("R = %.5e ", d.radius[1]))
    if d.sel[1] != 1
    else
        ev = findFrequency(get_manifold(mp), trs.p, d.omega)
        txt *= (@sprintf("Freq = %.5e ", abs(angle(ev)) / d.Tstep) *
                @sprintf("Damp = %.5e ", -log(abs(ev)) / abs(angle(ev))))
    end
    d.print(txt, repeat("\b", length(txt)), "\n")
    #     bson("LDIF-$(d.name).bson", Dict(:M => get_manifold(mp), :X => trs.p, :it => i))
    GC.gc(true)
end

mutable struct QPDebugARC <: DebugAction
    print::Function
    t0
    Tstep
    omega
    sel
    name
    QPDebugARC(Tstep, omega, sel, print::Function=print; d_name="data") = new(print, time(), Tstep, omega, sel, d_name)
end

# we need to calculate the actual frequency to check progress at least at the linear level
function (d::QPDebugARC)(mp, trs, i::Int)
    txt = ("        $(i). " *
           @sprintf("time = %.1f[s] ", time() - d.t0) *
           @sprintf("F(x) = %.5e ", get_cost(mp, trs.p)) *
           @sprintf("G(x) = %.5e ", norm(get_manifold(mp), trs.p, trs.X)))
    if d.sel[1] != 1
    else
        ev = findFrequency(get_manifold(mp), trs.p, d.omega)
        txt *= (@sprintf("Freq = %.5e ", abs(angle(ev)) / d.Tstep) *
                @sprintf("Damp = %.5e ", -log(abs(ev)) / abs(angle(ev))))
    end
    d.print(txt, repeat("\b", length(txt)), "\n")
    #     bson("LDIF-$(d.name).bson", Dict(:M => get_manifold(mp), :X => trs.p, :it => i))
    GC.gc(true)
end

# makes an array partition of the same size, but with ones(1) elements
@doc raw"""
    to_zero(x::ArrayPartition)

Creates an `ArrayPartition` the same structure as `x` expect that each component is a single element array with zero value.
"""
function to_zero(x::ArrayPartition)
    ArrayPartition(map(x -> x isa ArrayPartition ? to_zero(x) : zeros(1), x.x))
end

function to_bool(x::ArrayPartition)
    ArrayPartition(map(x -> x isa ArrayPartition ? to_bool(x) : [false], x.x))
end

function Optimise_parts(MF::QPCombinedFoliation, XF, thetaX, dataTX, thetaY, dataTY, sel;
    maxit=200, gradient_ratio=2^(-7), gradient_stop=2^(-29),
    cache=makeCache(MF, XF, thetaX, dataTX, thetaY, dataTY),
    omega=1.0, Tstep=1.0, radius=0.0, trust_region=true)
    Mp, Rp, VTp = Get_MRVT(MF, sel)
    Xp = Get_parts(XF, sel)
    # setting up scaling
    MK, XK = QPKernel(MF, XF)
    SetTorus!(MF, MK, XK)
    cache.pF .= QPScaleF(MK, XK, thetaX, dataTX)
    iit = 1
    G0 = Gradient(MF, XF, thetaX, dataTX, thetaY, dataTY, sel, cache=cache)
    if trust_region
        tr_max_radius = 4 * sqrt(manifold_dimension(Mp))
        if radius == 0.0
            tr_radius = tr_max_radius / 8
        else
            tr_radius = radius
        end
        tr_debug = QPDebugTRO(Tstep, omega, tr_radius, sel)
        XFres = trust_regions(Mp,
            (M, x) -> begin
                Set_parts!(XF, x, sel)
                updateCache!(cache, MF, XF, thetaX, dataTX, thetaY, dataTY, sel)
                Loss(MF, XF, thetaX, dataTX, thetaY, dataTY, cache=cache)
            end,
            (M, x) -> begin
                iit = 1
                Set_parts!(XF, x, sel)
                updateCache!(cache, MF, XF, thetaX, dataTX, thetaY, dataTY, sel)
                Gradient(MF, XF, thetaX, dataTX, thetaY, dataTY, sel, cache=cache)
            end,
            (M, x, dx) -> begin
                txt = @sprintf("H[%d]", iit)
                print(txt, repeat("\b", length(txt)))
                iit += 1
                Hessian(MF, XF, dx, thetaX, dataTX, thetaY, dataTY, sel, cache=cache)
            end,
            Xp,
            retraction_method=Rp,
            max_trust_region_radius=tr_max_radius,
            trust_region_radius=tr_radius,
            stopping_criterion=StopWhenAny(
                StopWhenGradientNormLess(max(norm(G0) * gradient_ratio, gradient_stop)),
                StopAfterIteration(maxit)
            ),
            debug=tr_debug
        )
        return XFres, tr_debug.radius[1]
    else
        arc_debug = QPDebugARC(Tstep, omega, sel)
        XFres = adaptive_regularization_with_cubics(Mp,
            (M, x) -> begin
                if (Get_parts(XF, sel) == x) && Get_parts(cache.Gvalid, sel)[1]
                else
                    Set_parts!(XF, x, sel)
                    Set_parts!(cache.Gvalid, false, sel)
                    updateCache!(cache, MF, XF, thetaX, dataTX, thetaY, dataTY, sel)
                end
                Loss(MF, XF, thetaX, dataTX, thetaY, dataTY, cache=cache)
            end,
            (M, x) -> begin
                if (Get_parts(XF, sel) == x) && Get_parts(cache.Gvalid, sel)[1]
                    return project(M, x, Get_parts(cache.DF, sel))
                else
                    iit = 1
                end
                Set_parts!(XF, x, sel)
                Set_parts!(cache.Gvalid, false, sel)
                updateCache!(cache, MF, XF, thetaX, dataTX, thetaY, dataTY, sel)
                Gradient(MF, XF, thetaX, dataTX, thetaY, dataTY, sel, cache=cache)
            end,
            (M, x, dx) -> begin
                txt = @sprintf("H[%d]", iit)
                print(txt, repeat("\b", length(txt)))
                iit += 1
                Hessian(MF, XF, dx, thetaX, dataTX, thetaY, dataTY, sel, cache=cache)
            end,
            Xp,
            retraction_method=Rp,
            stopping_criterion=StopWhenAny(
                StopWhenGradientNormLess(max(norm(G0) * gradient_ratio, gradient_stop)),
                StopAfterIteration(maxit)
            ),
            debug=arc_debug
        )
        return XFres
    end
end

@doc raw"""
    QPOptimise(MCF::QPCombinedFoliation, XCF, thetaTX, dataTX, thetaTY, dataTY;
        maxit=24, 
        gradient_ratio=2^(-7), gradient_stop=2^(-29), 
        steps=400, 
        name="default",
        cache=makeCache(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY),
        omega=1.0, Tstep=1.0, dataScale=1.0, dataId=1:size(thetaTX, 2), radii=to_zero(XCF), skipRS=false)
        
Use optimisation to fit the parameters of the combined foliation to data.

Input parameters are
* `MCF`, `XCF` the combined foliation
* `thetaTX`, `dataTX`, `thetaTY`, `dataTY` the data
* `maxit` maximum number of iterations within one trust-regions subproblem solution
* `gradient_ratio` the subproblem solution stops once the norm of the gradient is reduced by this factor
* `gradient_stop` the subproblem solution stops when the norm of the gradient is below this value
* `steps` the number of subproblem solutions as we iterate over the components of the combined foliation
* `name` the name of the problem, used for logging purposes
* `cache` is created by [`makeCache`](@ref)
* `omega` the forcing shift-angle
* `Tstep` the sampling period of the data ``\Delta t``
* `dataScale` the scaling factor used to scale the data
* `dataId` the indices of the data points from the original data set
* `radii` initial trust-regions radii for each component of the combined foliation
"""
function QPOptimise(MCF::QPCombinedFoliation, XCF, thetaTX, dataTX, thetaTY, dataTY;
                    maxit=24, gradient_ratio=2^(-7), gradient_stop=2^(-29), steps=400, name="default",
                    cache=makeCache(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY),
                    omega=1.0, Tstep=1.0, dataScale=1.0, dataId=1:size(thetaTX, 2), radii=to_zero(XCF), skipRS=false, trust_region=true)
    n_components = countParts(XCF)
    NGRAD = min(6, n_components) # repetition of gradient calculation
    NSEQ = 8 # steps + 2 # repetition of the sequential part -> does not ever happen

    t0 = time()
    XG = Gradient(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY, cache=cache)
    
    sel = Array{Any,1}(undef, 1)
    selSeq = Array{Any,1}(undef, 1)
    selCur = Array{Any,1}(undef, 1)
    sel[1] = (1,)
    selSeq[1] = (1,)
    selCur[1] = (1,)
    for k = 1:steps
        if mod(k, NGRAD) == NGRAD - 1
            print("|g")
            XG .= Gradient(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY, cache=cache)
        end
        #         sel[1], valM = maximumNorm(XG)
        while true
            if mod(k, NSEQ) == NSEQ - 1
                print("|s")
                # sequential component
                selSeq[1] = nextSel(XCF, selSeq[1])
                selCur[1] = selSeq[1]
            elseif (mod(k, NSEQ) == NSEQ - 2) && (sel[1] != (1,))
                print("|r")
                # R component is periodically updated
                selCur[1] = (1,)
            elseif (mod(k, NSEQ) == NSEQ - 3) && (sel[1] != (5,))
                print("|r")
                # S component is periodically updated
                selCur[1] = (5,)
            else
                selCur[1], valM = maximumNorm(XG)
                Set_parts!(XG, 0.0, selCur[1])
                print("|m")
            end
            if selCur[1] == sel[1] # if the same as previos iteration, skip
                print("|S")
                continue
            end
            # if the matrix is square it has full rank and therefore no need to optimise for
            if length(selCur[1]) > 1
                # a tensor component
                if (last(selCur[1]) != 1) && (length(unique(Get_part_size(XCF, selCur[1]))) == 1)
                    # not the root node AND matrix is square
                    print("|F")
                    continue
                end
            end
            if skipRS && ((selCur[1][1] == part_R) || (selCur[1][1] == part_S))
                print("|RS")
                continue
            end
            sel[1] = selCur[1]
            break
        end

        Set_parts!(XG, 0.0, sel[1])
        println(k, ". time=", time() - t0, " ", name, " sel=", sel[1])
        if trust_region
            _, radius = Optimise_parts(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY, sel[1],
                maxit=maxit, gradient_stop=gradient_stop, gradient_ratio=gradient_ratio, cache=cache,
                omega=omega, Tstep=Tstep, radius=Get_parts(radii, sel[1])[1], trust_region=trust_region)
            Set_parts!(radii, radius, sel[1])
        else
            Optimise_parts(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY, sel[1],
                maxit=maxit, gradient_stop=gradient_stop, gradient_ratio=gradient_ratio, cache=cache,
                omega=omega, Tstep=Tstep, radius=Get_parts(radii, sel[1])[1], trust_region=trust_region)
        end
        res, residU, residV = ResidualU(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY, cache=cache)
        println(@sprintf("Rmax = %.5e ", maximum(res)),
            @sprintf("Rmean = %.5e ", mean(res)),
            @sprintf("Rstd = %.5e ", std(res)),
            @sprintf("data = (U %d, V %d) / %d", length(findall(residU .> eps(1.0))),
                length(findall(residV .> eps(1.0))), length(res)))
        res, residU, residV = ResidualLoss(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY, cache=cache)
        println(@sprintf("Lmax = %.5e ", maximum(res)),
            @sprintf("Lmean = %.5e ", mean(res)),
            @sprintf("Lstd = %.5e ", std(res)),
            @sprintf("data = (U %d, V %d) / %d", length(findall(residU .> eps(1.0))),
                length(findall(residV .> eps(1.0))), length(res)))
        bson("CF-$(name).bson",
            Dict(:MCF => MCF, :XCF => XCF, :dataScale => dataScale, :Tstep => Tstep, :omega => omega, :dataId => dataId))
        GC.gc()
    end
end

@doc raw"""
    QPKernel(MF::QPCombinedFoliation, XF)
    
Returns the root of the combined foliation in the form of `MK`, `XK`, which is the same as the invariant torus.
"""
function QPKernel(MF::QPCombinedFoliation{dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order,ℝ}, XF) where {dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order}
    Q = zeros(eltype(QPC_U1_point(XF)), dim_data, dim_data, 2 * fourier_order + 1)
    Q[1:dim_latent, :, :] .= permutedims(QPC_U1_point(XF), (2, 1, 3))
    Q[dim_latent+1:end, :, :] .= permutedims(QPC_V1_point(XF), (2, 1, 3))

    grid = getgrid(fourier_order)
    MK = QPConstant(dim_data,fourier_order)
    XK = zero(MK)
    XK0 = zero(MK)
    
    for it in range(1,64)
        for k in eachindex(grid)
            XK[:,k] .= - Q[:,:,k] \ vcat(QPC_U0_point(XF)[:,k] .+ EvalUnl(MF, XF, grid[k], XK0[:,k]), QPC_V0_point(XF)[:,k] .+ EvalVnl(MF, XF, grid[k], XK0[:,k]))
        end
        nm = norm(XK0 - XK)
        XK0 .= XK
        if nm <= eps(sum(XK0 .^ 2))
            println("KERNEL = ", norm(XK), " it = ", it)
            break
        end
    end
    return MK, XK
end

# Reconstruct the invariant manifold from Foliation and LocalFoliation
#   z = U o W
#   0 = Ut * W
# ----
#   z = U0  +  U1 W + Unl o W
#   0 = Ut0 + Ut1 W + Wt
# Rearrange, and use the linear part to solve for the rest
#   U1 W  = z - U0  - Unl o W
#   Ut1 W =   - Ut0 - Wt
# Create the matrix
#   Q = (U1 Ut1)^T
# Then
#   W = Q^-1 (z - U0  - Unl o W )
#            (  - Ut0 - Wt)
function QPReconstruct(MF::QPCombinedFoliation{dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order,ℝ}, XF) where {dim_data,dim_latent,fourier_order,R_order,U_order,S_order,V_order}
    Q = zeros(eltype(QPC_U1_point(XF)), dim_data, dim_data, 2 * fourier_order + 1)
    Q[1:dim_latent, :, :] .= permutedims(QPC_U1_point(XF), (2, 1, 3))
    Q[dim_latent+1:end, :, :] .= permutedims(QPC_V1_point(XF), (2, 1, 3))
    #     Q[1:dim_latent, 1:dim_latent, :] .= Array(I,dim_latent,dim_latent)
    #     Q[1:dim_latent, dim_latent+1:end, :] .= QPC_U1_point(XF)
    #     Q[dim_latent+1:end, 1:dim_latent, :] .= QPC_V1_point(XF)
    #     Q[dim_latent+1:end, dim_latent+1:end, :] .= Array(I,dim_data-dim_latent,dim_data-dim_latent)
    # the identity part
    ID0 = zero(Q)
    for k = 1:dim_latent
        ID0[k, k, :] .= one(eltype(ID0))
    end

    # the immersion
    MW = QPPolynomial(dim_data, dim_latent, fourier_order, 0, max(U_order, V_order))
    XW0 = zero(MW)
    XW = zero(MW)

    # the full polynomial
    MUoW = QPPolynomial(dim_data, dim_latent, fourier_order, 0, max(U_order, V_order))
    XUoW = zero(MUoW)
    XUnloW = zero(MUoW)
    #test
    #     @show EvalUnl(MF, XF, 1.0, randn(dim_data))
    #     @show EvalUnl(MF, XF, 1.0, Eval(MW, XW, 1.0, randn(dim_latent)))
    #     @show Eval(QPL_W_manifold(MLF), QPL_W_point(XLF), 1.0, randn(dim_latent))

    while true
        # setting to zero
        XUoW .= zero(eltype(XUoW))
        # minus constant
        SetConstantPart!(MUoW, XUoW, -vcat(QPC_U0_point(XF), QPC_V0_point(XF)))
        # plus identity
        SetLinearPart!(MUoW, XUoW, ID0)
        # calculate nonlinear
        fromFunction!(MUoW, XUnloW,
            (x, t) -> vcat(
                EvalUnl(MF, XF, t, Eval(MW, XW, t, x)),
                EvalVnl(MF, XF, t, Eval(MW, XW, t, x))))
        # minus nonlinear
        XUoW .-= XUnloW
        # next iteration
        for k = 1:size(XUoW, 3)
            XW0[:, :, k] .= Q[:, :, k] \ XUoW[:, :, k]
        end
        nm = norm(XW0 - XW)
        XW .= XW0
        if nm <= eps(sum(XW0 .^ 2))
            break
        end
    end

    return MW, XW
end

@doc raw"""
    QPPostProcess(MCF::QPCombinedFoliation, XCF, omega, resonance=false, threshold=0.1)
    
Extracts the invariant manifold from the combined foliation together with its dynamics. The result is in a normal form with mixed power series Fourier series form.

Paramaters:
* `MCF`, `XCF` the combined foliation.
* `W1` the inverse trabsformation produced by [`QPPreProcess`](@ref)
* `omega` the forcing shift-angle.
* `resonance` `true` if to account for non-autonomous internal resonances. When `false`, the result will be an autonomous normal form.
* `threshold` if the distance between the two side of the non-resonance criterion is less than `threshold`, it counts as an internal (or external) resonance

Returns:
```
MSn, XSn, MFW, XFWoWdoWn, MW, XW
```
* `MSn`, `XSn` the conjugate dynamics on the manifold in normal form
* `MFW`, `XFWoWdoWn` manifold immersion in the physical coordinate system and adapted to correspond to the normal form
* `MW`, `XW` manifold immersion in the coordinate system of the combined foliation (not physical coordinates)
"""
function QPPostProcess(MCF::QPCombinedFoliation, XCF, W1, omega, resonance=false, threshold=0.1)
    MW, XW0 = QPReconstruct(MCF, XCF)
    @tullio XW[i,j,k] := W1[i,p,k] * XW0[p,j,k]
    Lambda, MWd, XWd, MUd, XUd, MUFW, XUFW, XWre = smoothestDecomposition(QPC_R_manifold(MCF), QPC_R_point(XCF), omega)
    MFWd, XFWd = toFourier(MWd, XWd)
    MSd, XSd = toFourier(MUFW, XUFW)

    # STEP 6
    # Create the normal form
    #     MSn, XSn, MUn, XUn = iQPFoliation(MSd, XSd, omega, [1; 2], resonance = false)
    MSn, XSn, MWn, XWn = iQPManifold(MSd, XSd, omega, [1; 2], resonance=resonance, threshold=threshold)

    # STEP 7
    # Compose all the immersions
    MFW, XFW = toFourier(MW, XW)
    XFWoWd = zero(MFW)
    XFWoWdoWn = zero(MFW)
    QPFourierPolynomialSubstitute!(MFW, XFWoWd, MFW, XFW, MFWd, XFWd)
    QPFourierPolynomialSubstitute!(MFW, XFWoWdoWn, MFW, XFWoWd, MWn, XWn)
    return MSn, XSn, MFW, XFWoWdoWn, MW, XW
end


function testQPCombined()
    omega = 1.0
    npt = 100
    dim_data = 5
    dim_latent = 2
    fourier_order = 3
    R_order = 5
    U_order = 5
    V_order = 5
    S_order = 2
    grid = getgrid(fourier_order)
    dataUX = randn(dim_latent, npt) / 8
    dataUY = randn(dim_latent, npt) / 8
    dataVX = randn(dim_data - dim_latent, npt) / 8
    dataVY = randn(dim_data - dim_latent, npt) / 8
    dataTX = vcat(dataUX, dataVX)
    dataTY = vcat(dataUY, dataVY)
    theta = 2 * pi * rand(npt)
    thetaX = fourierInterplate(grid, theta)
    thetaY = fourierInterplate(grid, theta .+ omega)

    # these do not matter much, they will be refactored
    S1_start = randn(dim_latent, dim_latent, 2 * fourier_order + 1)
    U0_start = randn(dim_latent, 2 * fourier_order + 1)
    U1_start = randn(dim_latent, dim_data, 2 * fourier_order + 1)
    B_start = randn(dim_data - dim_latent, dim_data - dim_latent, 2 * fourier_order + 1)
    V0_start = randn(dim_data - dim_latent, 2 * fourier_order + 1)
    V1_start = randn(dim_data - dim_latent, dim_data, 2 * fourier_order + 1)
    MCF, XCF = QPCombinedFoliation(dim_data, dim_latent, fourier_order, R_order, U_order, S_order, V_order, S1_start, B_start, sparse=false)
    XCF = randn(MCF)
    @show QPC_R_point(XCF)
    @show QPC_S_point(XCF)

    A = GetLinearPart(QPC_R_manifold(MCF), QPC_R_point(XCF))
    #     QPC_R_point(XCF) ./= 8
    #     QPC_S_point(XCF) .= 0
    SetLinearPart!(QPC_R_manifold(MCF), QPC_R_point(XCF), A)
    #     QPC_U0_point(XCF) ./= 8
    #     QPC_B_point(XCF) ./= 8
    #     QPC_V0_point(XCF) ./= 8
    #     QPC_Vnl_point(XCF) ./= 8

    cache = makeCache(MCF, XCF, thetaX, dataTX, thetaY, dataTY)
    @show Loss(MCF, XCF, thetaX, dataTX, thetaY, dataTY, cache=cache)

    sel = Array{Any,1}(undef, 1)
    sel[1] = (1,)

    l_loss = (M, x) -> begin
        #                 print("L ")
        Set_parts!(XCF, x, sel[1])
        updateCache!(cache, MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1])
        Loss(MCF, XCF, thetaX, dataTX, thetaY, dataTY, cache=cache)
    end
    l_grad = (M, x) -> begin
        Set_parts!(XCF, x, sel[1])
        updateCache!(cache, MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1])
        Gradient(MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1], cache=cache)
    end
    l_hess = (M, x, dx_) -> begin
        Set_parts!(XCF, x, sel[1])
        updateCache!(cache, MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1])
        _ = Gradient(MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1], cache=cache)
        Hessian(MCF, XCF, dx_, thetaX, dataTX, thetaY, dataTY, sel[1], cache=cache)
    end

    epsilon = 2^(-22)
    XCFD = deepcopy(XCF)
    println("GRADIENTs")
    while true
        println("PART = ", sel[1])
        # part_S
        Msel, Rsel, Vsel = Get_MRVT(MCF, sel[1])
        Xsel = deepcopy(Get_parts(XCF, sel[1]))
        @show size(Xsel)
        pl = check_gradient(Msel, l_loss, l_grad, Xsel, randn(Msel), retraction_method = Rsel, limits = (-8, 1), plot = true, io = stdout)
        display(pl)
        read(stdin, Char)
        println("Calculation time")
        @time updateCache!(cache, MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1])
        @time grad = Gradient(MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1]; riemannian = false, cache = cache)
        gradFD = deepcopy(grad)
        XCFp = Get_parts(XCF, sel[1])
        XCFDp = Get_parts(XCFD, sel[1])
        for k in eachindex(gradFD)
            XCFDp[k] += epsilon
            updateCache!(cache, MCF, XCFD, thetaX, dataTX, thetaY, dataTY, sel[1])
            gradFD[k] = Loss(MCF, XCFD, thetaX, dataTX, thetaY, dataTY, cache = cache) 
            updateCache!(cache, MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1])
            gradFD[k] -= Loss(MCF, XCF, thetaX, dataTX, thetaY, dataTY, cache = cache)
            gradFD[k] /= epsilon
            XCFDp[k] = XCFp[k]
        end
        @show maximum(abs.(gradFD - grad)), maximum(abs.((gradFD - grad) ./ gradFD)), maximum(abs.(gradFD)), maximum(abs.(grad))
        
#         Optimise_parts(MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1],
#                         maxit = 10, cache = cache, omega = omega, Tstep = 1.0)
        
        sel[1] = nextSel(XCF, sel[1])
        while (length(sel[1]) > 1) && (last(sel[1]) != 1) && (length(unique(Get_part_size(XCF, sel[1]))) == 1)
            sel[1] = nextSel(XCF, sel[1])
        end
        if sel[1] == (1,)
            break
        end
    end

    println("HESSIANs")
    sel[1] = (1,)
    while true
        println("PART = ", sel[1])
        # part_S
        XCFp = Get_parts(XCF, sel[1])
        XCFDp = Get_parts(XCFD, sel[1])
        dx = randn(size(XCFp)...)

        Msel, Rsel, Vsel = Get_MRVT(MCF, sel[1])
        Xsel = deepcopy(Get_parts(XCF, sel[1]))
        # check_Hessian(M,       f,      grad_f, Hess_f, p=rand(M), X=rand(M; vector_at=p), Y=rand(M, vector_at=p); kwargs...)
        pl = check_Hessian(Msel, l_loss, l_grad, l_hess, Xsel, randn(Msel, Xsel), randn(Msel, Xsel); check_grad=false, retraction_method=Rsel, limits=(-8, 1), plot=true, io=stdout, atol=1e-10, rtol=1e-6, exactness_tol=1e-10)
        display(pl)
        read(stdin, Char)

        println("Calculation time")
        @time updateCache!(cache, MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1])
        @time hess = Hessian(MCF, XCF, dx, thetaX, dataTX, thetaY, dataTY, sel[1]; riemannian=false, cache=cache)
        hessFD = deepcopy(hess)
        #         @show hess
        for k in eachindex(hessFD)
            XCFDp[k] += epsilon
            updateCache!(cache, MCF, XCFD, thetaX, dataTX, thetaY, dataTY, sel[1])
            #             G = Gradient(MCF, XCFD, thetaX, dataTX, thetaY, dataTY, sel[1]; riemannian = false)
            G = Gradient(MCF, XCFD, thetaX, dataTX, thetaY, dataTY, sel[1]; riemannian=false, cache=cache)
            updateCache!(cache, MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1])
            #             G -= Gradient(MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1]; riemannian = false)
            G -= Gradient(MCF, XCF, thetaX, dataTX, thetaY, dataTY, sel[1]; riemannian=false, cache=cache)
            hessFD[k] = sum(G .* dx) / epsilon
            XCFDp[k] = XCFp[k]
        end
        #         @show hessFD - hess
        @show maximum(abs.(hessFD - hess)), maximum(abs.((hessFD - hess) ./ hessFD)), maximum(abs.(hessFD))
        sel[1] = nextSel(XCF, sel[1])
        while (length(sel[1]) > 1) && (last(sel[1]) != 1) && (length(unique(Get_part_size(XCF, sel[1]))) == 1)
            sel[1] = nextSel(XCF, sel[1])
        end
        if sel[1] == (1,)
            break
        end
    end
end
