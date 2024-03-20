
function ComplexReal2(fourier_order)
    M2 = QPFourierPolynomial(2, 2, fourier_order, 1, 1)
    X2R = zero(M2)
    X2RL = GetLinearPart(M2, X2R)
    T2R = convert.(eltype(X2R), [1; 1im;; 1; -1im] / sqrt(2))
    T2C = inv(T2R)
    X2RL[:, :, fourier_order+1] .= T2R
    SetLinearPart!(M2, X2R, X2RL)
    X2C = zero(M2)
    X2CL = GetLinearPart(M2, X2C)
    T2C = inv(T2R)
    X2CL[:, :, fourier_order+1] .= T2C
    SetLinearPart!(M2, X2C, X2CL)
    return M2, X2R, X2C
end

# MSf, XSf -> S
# MUf, XUf -> foliation decomposition
# MUd, XUd -> linear bundle decomposition
# output:
# U = R . Uf . Ud
# S = R . S . R^-1
@doc raw"""
    FoliationToReal(MSf::QPFourierPolynomial, XSf, MUf, XUf, MUd, XUd)
    
Input
* `MSf`, `XSf` the conjugate map in complex coordinates, Fourier ``\boldsymbol{S}``
* `MUf`, `XUf` the immersion from the modal coordinate system, Fourier ``\boldsymbol{U}``
* `MUd`, `XUd` the transformation from physical to modal coordinate system ``\boldsymbol{U}_d``
The invariance equation is
```math
    \boldsymbol{S} \circ \boldsymbol{U} \circ \boldsymbol{U}_d = \boldsymbol{U} \circ \boldsymbol{U}_d \circ \boldsymbol{F}
```

Let us denote ``\boldsymbol{R}_2`` the complex to real transformation, ``\boldsymbol{C}_2``, the real to complex transformation, such that  ``\boldsymbol{C}_2 \circ ``\boldsymbol{R}_2 = \boldsymbol{I}`` We then calculate
```math
\begin{aligned}
    \boldsymbol{S}_r &= \boldsymbol{R}_2 \circ \boldsymbol{S} \circ \boldsymbol{C}_2 \\
    \boldsymbol{U}_r &= \boldsymbol{R}_2 \circ \boldsymbol{U} \circ \boldsymbol{U}_d
```
These are converted to pointwise representation on the grid.

Return values:
```
MSr, XSr, MUr, XUr,
```
where
* `MSr`, `XSr` is ``\boldsymbol{S}_r``
* `MUr`, `XUr` is ``\boldsymbol{U}_r``
"""
function FoliationToReal(MSf::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, XSf, MUf, XUf, MUd, XUd) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    MUFd, XUFd = toFourier(MUd, XUd)
    XUfoUt = zero(MUf)
    QPFourierPolynomialSubstitute!(MUf, XUfoUt, MUf, XUf, MUFd, XUFd)
    M2, X2R, X2C = ComplexReal2(fourier_order)

    XRoUfoUt = zero(MUf)
    QPFourierPolynomialSubstitute!(MUf, XRoUfoUt, M2, X2R, MUf, XUfoUt)
    # S ->
    XRoS = zero(MSf)
    QPFourierPolynomialSubstitute!(MSf, XRoS, M2, X2R, MSf, XSf)
    XSt = zero(MSf)
    QPFourierPolynomialSubstitute!(MSf, XSt, MSf, XRoS, M2, X2C)

    MUr, XUr = fromFourier(MUf, XRoUfoUt)
    MSr, XSr = fromFourier(MSf, XSt)
    return MSr, XSr, MUr, XUr, MSf, XSt
end

function FoliationToOriginal(MUf::QPFourierPolynomial, XUf, MUd, XUd)
    MU, XU = fromFourierComplex(MUf, XUf)
    XUoUd = zero(MU)
    QPPolynomialSubstitute!(MU, XUoUd, MU, XU, MUd, XUd)
    return MU, XUoUd
end

# MSf, XSf -> S
# MUf, XUf -> foliation decomposition
# MUd, XUd -> linear bundle decomposition
# output:
# U = R . Uf . Ud
# S = R . S . R^-1
function FoliationToReal(MSf::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, XSf, MUf, XUf) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    M2, X2R, X2C = ComplexReal2(fourier_order)

    XRoUfoUt = zero(MUf)
    QPFourierPolynomialSubstitute!(MUf, XRoUfoUt, M2, X2R, MUf, XUf)
    # S ->
    XRoS = zero(MSf)
    QPFourierPolynomialSubstitute!(MSf, XRoS, M2, X2R, MSf, XSf)
    XSt = zero(MSf)
    QPFourierPolynomialSubstitute!(MSf, XSt, MSf, XRoS, M2, X2C)

    MUr, XUr = fromFourier(MUf, XRoUfoUt)
    MSr, XSr = fromFourier(MSf, XSt)
    return MSr, XSr, MUr, XUr
end

# plugging in X2C into the polynomial
function ManifoldToComplex(MW::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, XW) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    M2, X2R, X2C = ComplexReal2(fourier_order)
    XWc = zero(MW)
    QPFourierPolynomialSubstitute!(MW, XWc, MW, XW, M2, X2C)
    return MW, XWc
end

#------------------------------------------------------------------------------------------------------------------------------------------
#
# direct FOLIATION from a map
#
#------------------------------------------------------------------------------------------------------------------------------------------

# input: 
#   M, X, omega represent a system that is transformed into a diagonal form.
#       It is assumed that the linear part of the system is made autonomous
#   sel_ is the set of eigenvalues we need
#   resonance: 
#       true of internal resonances with non-zero frequencies are allowed
#       false (default) if we want the output be completely autonomous (even if it blows up)
# output:
#   MS, XS, MU, XU
#       MS, XS are the reduced dynamics
#       MU, XU are the submersion
function iQPFoliation(M::QPFourierPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}, X, omega, sel_; resonance::Bool=true, threshold=0.1, rom_order=max_polyorder) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    sel = vec(sel_)
    # co-dimension of the foliation
    f_dim = length(sel)
    # create the output
    MU = QPFourierPolynomial(f_dim, dim_in, fourier_order, min_polyorder, max_polyorder)
    XU = zero(MU)
    # create ROM
    MS = QPFourierPolynomial(f_dim, f_dim, fourier_order, min_polyorder, max_polyorder)
    XS = zero(MS)
    # We are solving the invariance equation
    #   S(q, U(t, x)) = U(t+omega, F(x))
    # it is assumed that LP is diagonal and only has a constant part
    LP = GetLinearPart(M, X)
    # Lambda is an averaged linear part
    @show Lambda = diag(LP[:, :, fourier_order+1])
    # clearing the time-dependent parts of M, X
    LP .= zero(eltype(LP))
    LP[:, :, fourier_order+1] .= Diagonal(Lambda)
    SetLinearPart!(M, X, LP)
    # Clearing the constant part
    SetConstantPart!(M, X, zero(eltype(X)))
    # setting the linear part
    LU = GetLinearPart(MU, XU)
    #     @show size(LU[:,sel,fourier_order+1])
    LU[:, sel, fourier_order+1] .= one(LU[:, sel, fourier_order+1])
    SetLinearPart!(MU, XU, LU)
    #
    LS = GetLinearPart(MS, XS)
    LS[:, :, fourier_order+1] .= Diagonal(Lambda[sel])
    SetLinearPart!(MS, XS, LS)
    # the residual, after subs R(t,x) = S(q, U_k(t, x)) - U_k(t+omega, F(x)) 
    # same polynomial as MU
    XR = zero(MU)
    # the Fourier representetaion starts with Exp(i * (- fourier_order) * t) as per convention
    # so shifting U(t+omega,.) is the same as multiplying with Exp(i * k * omega) ... k = - fourier_order : fourier_order

    # create the residual
    #   R(t,x) = S(q, U_k(t, x)) - U_k(t+omega, F(x))
    X_SoU = zero(MU)
    tabs_SoU = QPFourierSubstituteTabs(MU, X_SoU, MS, XS, MU, XU)
    QPFourierPolynomialSubstitute!(MU, X_SoU, MS, XS, MU, XU, tabs_SoU...)
    X_UoF = zero(MU)
    tabs_UoF = QPFourierSubstituteTabs(MU, X_UoF, MU, XU, M, X)
    QPFourierPolynomialSubstitute!(MU, X_UoF, MU, XU, M, X, tabs_UoF..., shift=omega)
    XR .= X_UoF .- X_SoU
    #     mx = findall(abs.(GetLinearPart(MU, X_SoU)) .> 1e-6)

    for ord = 2:PolyOrder(MU)
        id = PolyOrderIndices(MU, ord)
        # within this, check each term for each dimension and Fourier order
        for ij in id
            Sij = PolyFindIndex(MS.mexp, MU.mexp[sel, ij])
            for i0 = 1:size(XU, 1), kp = 1:size(XU, 3)
                k = kp - 1 - fourier_order
                den = Lambda[sel[i0]] - prod(Lambda .^ MU.mexp[:, ij]) * exp(1im * k * omega)
                denAngle = angle(Lambda[sel[i0]]) - sum(angle.(Lambda) .* MU.mexp[:, ij]) + k * omega
                if ord == sum(MU.mexp[sel, ij])
                    # internal resonances
                    if (abs(denAngle) > threshold) || (ord > rom_order)
                        # no internal resonance
                        XS[i0, Sij, kp] = zero(den)
                        XU[i0, ij, kp] = XR[i0, ij, kp] / den
                    else
                        # internal resonance
                        if ((k == 0) || resonance)
                            # makes it autonomous
                            XS[i0, Sij, kp] = XR[i0, ij, kp]
                            XU[i0, ij, kp] = zero(den)
                            #                             println("*** Internal: ", abs(den), " i0=", i0, " ij=", MU.mexp[:,ij], " k=", k, " Sij=", MS.mexp[:,Sij])
                            println("*** Internal: ", @sprintf("abs=%.5e", abs(den)), " i0=", i0, " ij=", MU.mexp[:, ij], " k=", k, " Sij=", MS.mexp[:, Sij], @sprintf(" angle=%.5e", abs(denAngle)))
                        else
                            XS[i0, Sij, kp] = zero(den)
                            XU[i0, ij, kp] = XR[i0, ij, kp] / den
                            #                             println("---  Internal unmitigated: ", abs(den), " i0=", i0, " ij=", MU.mexp[:,ij], " k=", k, " Sij=", MS.mexp[:,Sij])
                            println("---  Internal unmitigated: ", @sprintf("abs=%.5e", abs(den)), " i0=", i0, " ij=", MU.mexp[:, ij], " k=", k)
                        end
                    end
                else
                    XU[i0, ij, kp] = XR[i0, ij, kp] / den
                    if abs(den) < threshold
                        # unmitigated resonance
                        println("External: ", @sprintf("abs=%.5e", abs(den)), " i0=", i0, " ij=", MU.mexp[:, ij], " k=", k)
                    end
                end
            end
        end
        # S o U
        QPFourierPolynomialSubstitute!(MU, X_SoU, MS, XS, MU, XU, tabs_SoU...)
        # U o F
        QPFourierPolynomialSubstitute!(MU, X_UoF, MU, XU, M, X, tabs_UoF..., shift=omega)
        XR .= X_UoF .- X_SoU
    end
    #     @show abs.(XR)
    @show maximum(abs.(XR))
    return MS, XS, MU, XU
end

function iQPReconstruct(MU::QPPolynomial{dim_latent,dim_data,fourier_order,min_polyorder1,max_polyorder1,field}, XU,
    MV::QPPolynomial{dim_transverse,dim_data,fourier_order,min_polyorder2,max_polyorder2,field}, XV) where
{dim_latent,dim_data,fourier_order,min_polyorder1,max_polyorder1,dim_transverse,min_polyorder2,max_polyorder2,field}
    Q = zeros(eltype(XU), dim_data, dim_data, 2 * fourier_order + 1)
    Q[1:dim_latent, :, :] .= GetLinearPart(MU, XU)
    Q[dim_latent+1:end, :, :] .= GetLinearPart(MV, XV)
    # the identity part
    ID0 = zero(Q)
    for k = 1:dim_latent
        ID0[k, k, :] .= one(eltype(ID0))
    end

    # the immersion
    MW = QPPolynomial(dim_data, dim_latent, fourier_order, 0, max(max_polyorder1, max_polyorder2), field)
    XW0 = zero(MW)
    XW = zero(MW)

    # the full polynomial
    MUoW = QPPolynomial(dim_data, dim_latent, fourier_order, 0, max(max_polyorder1, max_polyorder2), field)
    XUoW = zero(MUoW)
    XUnloW = zero(MUoW)
    #test
    #     @show EvalUnl(MF, XF, 1.0, randn(dim_data))
    #     @show EvalUnl(MF, XF, 1.0, Eval(MW, XW, 1.0, randn(dim_latent)))
    #     @show Eval(QPL_W_manifold(MLF), QPL_W_point(XLF), 1.0, randn(dim_latent))
    XUc = deepcopy(XU)
    XVc = deepcopy(XV)
    if max(min_polyorder1, min_polyorder2) == 0
        SetConstantPart!(MU, XUc, zero(eltype(XUc)))
        SetConstantPart!(MV, XVc, zero(eltype(XVc)))
    end
    SetLinearPart!(MU, XUc, zero(eltype(XUc)))
    SetLinearPart!(MV, XVc, zero(eltype(XVc)))
    for k = 1:max(max_polyorder1, max_polyorder2)+1
        # setting to zero
        XUoW .= zero(eltype(XUoW))
        # minus constant
        if max(min_polyorder1, min_polyorder2) == 0
            SetConstantPart!(MUoW, XUoW, -vcat(GetConstantPart(MU, XUc), GetConstantPart(MV, XVc)))
        end
        # plus identity
        SetLinearPart!(MUoW, XUoW, ID0)
        # calculate nonlinear
        fromFunction!(MUoW, XUnloW,
            (x, t) -> vcat(
                Eval(MU, XUc, t, Eval(MW, XW, t, x)),
                Eval(MV, XVc, t, Eval(MW, XW, t, x))))
        # minus nonlinear
        XUoW .-= XUnloW
        # next iteration
        for l = 1:size(XUoW, 3)
            XW0[:, :, l] .= Q[:, :, l] \ XUoW[:, :, l]
        end
        @show nm = norm(XW0 - XW)
        XW .= XW0
        if nm <= eps(norm(XW0)^2)
            break
        end
    end
    # check the reconstruction
    MUoW = QPPolynomial(dim_latent, dim_latent, fourier_order, 0, max(max_polyorder1, max_polyorder2), field)
    XUoW = zero(MUoW)
    MVoW = QPPolynomial(dim_transverse, dim_latent, fourier_order, 0, max(max_polyorder1, max_polyorder2), field)
    XVoW = zero(MVoW)
    QPPolynomialSubstitute!(MUoW, XUoW, MU, XU, MW, XW)
    QPPolynomialSubstitute!(MVoW, XVoW, MV, XV, MW, XW)
    @show LU = GetLinearPart(MUoW, XUoW)
    SetLinearPart!(MUoW, XUoW, zero(LU))
    @show maximum(abs.(XUoW)), maximum(abs.(XVoW))
    return MW, XW
end

# INPUT
#   MFd, XFd : the map in the coordinate system of MUd, XUd
#   MUd, XUd : the map turning old coodinates into new ones. 
#       required to transform the result into the original coordinates for MW, XW
@doc raw"""
    QPGraphStyleFoliations(MP::QPPolynomial, XP, omega, SEL; dataScale=1, resonance=false, threshold=0.1, rom_order=max_polyorder)

The purpose of this function is to calculate two complementary invariant foliations about an invariant torus and then extract the invariant manifold that is the zero level set of the second foliation. The invariant manifold has the same dynamics as the first foliation.
    
The inputs are the following
* `MP`, `XP` the Poincare map that describes the dynamics.
* `omega` the phase shift at each time-step
* `SEL` which spectral bundles to use for the calculation of the invariant manifold
* `dataScale` is the scaling factor that scales the coordinate system of the invariant manifold. This scaling is used when we would like to compare the invariant manifold identified from data that has been scaled. Using the same scaling factor makes this comparison possible.
* `threshold` if a near resonance is greater than this value, it is ignored
* `resonance`: when `false` the conjugate map is reduced to an autonomous system, even if there are near resonances. 
    Otherwise all near resonances are accounted for in the conjugate map.
* `rom_order` what is the highest order for which near resonances are taken into account. It defaults to the order of the Poincare map `MP`, `XP`.

Return values are:
```
MRf, XRf, MW, XW, MRt, XRt, MUV, XUVflat, MSt, XSt, MVU, XVUflat
```
* `MRf`, `XRf` the conjugate dynamics of the first invariant foliation (`SEL`) in normal form coordinates and with Fourier coefficients
* `MW`, `XW` the immersion of the invariant manifold in the original coordinate system

The outputs that could be used to set initial conditions to a foliation
* `MRt`, `XRt` the conjugate dynamics of the first invariant foliation (`SEL`) when the foliation is parametrised graph-style
* `MUV`, `XUVflat`
* `MSt`, `XSt` the conjugate dynamics of the second invariant foliation when the foliation is parametrised graph-style
* `MVU`, `XVUflat`
"""
function QPGraphStyleFoliations(MP::QPPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,ℝ}, XPmap, omega, SEL; dataScale=1, resonance=false, threshold=0.1, rom_order=max_polyorder) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder}
    MK = QPConstant(dim_out, fourier_order)
    XK = zero(MK)
    torusId!(MK, XK, MP, XPmap, omega)

    canMap = (x, t) -> Eval(MP, XPmap, t, Eval(MK, XK, t) + x) - Eval(MK, XK, t + omega)
    XP0 = fromFunction(MP, canMap)

    # the linear transformation
    Lambda, MWd, XWd, MUd, XUd, MUFW, XUFW, XWre = smoothestDecomposition(MP, XP0, omega)
    MFd, XFd = toFourier(MUFW, XUFW) # in the coordinate system of XWd (inverse transf), XUd (forward transf)

    MRf, XRf, MUf, XUf = iQPFoliation(MFd, XFd, omega, SEL, resonance=resonance, threshold=threshold, rom_order=rom_order)
    # foliation of the approximate derived map TRANSVERSAL
    MSf, XSf, MVf, XVf = iQPFoliation(MFd, XFd, omega, setdiff(1:dim_out, SEL), resonance=resonance, threshold=threshold, rom_order=rom_order)
    # putting back to the original frame
    MU, XU = FoliationToOriginal(MUf, XUf, MUd, XUd)
    MV, XV = FoliationToOriginal(MVf, XVf, MUd, XUd)
    # reconstructing the invariant manifold
    MW, XW = iQPReconstruct(MU, XU, MV, XV)
    # putting back into Fourier form
    MWf, XWf = toFourier(MW, XW)
    # checking if the manifold is invariant in the original coordinate system with real coordinates
#     XWcoS = zero(MWf)
#     XWcoF = zero(MWf)
#     MFf, XFf = toFourier(MP, XP0)
#     QPFourierPolynomialSubstitute!(MWf, XWcoS, MWf, XWf, MRf, XRf, shift = omega)
#     QPFourierPolynomialSubstitute!(MWf, XWcoF, MFf, XFf, MWf, XWf)
#     XR = XWcoS - XWcoF
#     println("Combined Foliation Residual")
#     @show maximum(abs.(XR))
#     @show GetLinearPart(MWf, XR)
    
    # MRr, XRr, MUr, XUr -> real and physical coordinates, pointwise
    # MSr, XSr, MVr, XVr -> real and physical coordinates, pointwise
    # it is not used from this point onwards!
    MRr, XRr, MUr, XUr = FoliationToReal(MRf, XRf, MUf, XUf, MUd, XUd)
    MSr, XSr, MVr, XVr = FoliationToReal(MSf, XSf, MVf, XVf, MUd, XUd)

    U1 = GetLinearPart(MUr, XUr)
    V1 = GetLinearPart(MVr, XVr)
    TRi = vcat(U1, V1)
    TR = zero(TRi)
    for k in axes(TR, 3)
        TR[:, :, k] .= inv(TRi[:, :, k]) .* dataScale
    end
    # transformation TR diagonalises the system and takes into account data scaling
    MTR = QPPolynomial(dim_out, dim_out, fourier_order, 1, 1)
    XTR = zero(MTR)
    SetLinearPart!(MTR, XTR, TR)
    XUtr = zero(MUr)
    XVtr = zero(MVr)
    QPPolynomialSubstitute!(MUr, XUtr, MUr, XUr, MTR, XTR)
    QPPolynomialSubstitute!(MVr, XVtr, MVr, XVr, MTR, XTR)
    
    # We are creating MUU, XUU and MVV, XVV that must be linear eventually
    WU1 = zeros(dim_out, length(SEL))
    WU1[1:length(SEL),:] .= Array(I, length(SEL), length(SEL))
    WV1 = zeros(dim_out, dim_out - length(SEL))
    WV1[length(SEL)+1:end,:] .= Array(I, dim_out - length(SEL), dim_out - length(SEL))
    MWU = QPPolynomial(dim_out, length(SEL), fourier_order, 1, 1)
    MWV = QPPolynomial(dim_out, dim_out - length(SEL), fourier_order, 1, 1)
    XWU = zero(MWU)
    XWV = zero(MWV)
    SetLinearPart!(MWU, XWU, WU1)
    SetLinearPart!(MWV, XWV, WV1)
    # the transformed parts
    MUU = QPPolynomial(length(SEL), length(SEL), fourier_order, 1, max_polyorder)
    MVV = QPPolynomial(dim_out - length(SEL), dim_out - length(SEL), fourier_order, 1, max_polyorder)
    XUU = zero(MUU)
    XVV = zero(MVV)
    QPPolynomialSubstitute!(MUU, XUU, MUr, XUtr, MWU, XWU)
    QPPolynomialSubstitute!(MVV, XVV, MVr, XVtr, MWV, XWV)
    # MUU, XUU and MVV, XVV are 
    
    # make the inverse transformation...
    @show GetLinearPart(MUU, XUU)
    XUUinv = inverse(MUU, XUU)
    @show GetLinearPart(MVV, XVV)
    XVVinv = inverse(MVV, XVV)
    XUtr_flat = zero(MUr)
    XVtr_flat = zero(MVr)
    QPPolynomialSubstitute!(MUr, XUtr_flat, MUU, XUUinv, MUr, XUtr)
    QPPolynomialSubstitute!(MVr, XVtr_flat, MUU, XUUinv, MVr, XVtr)

    MRt, XRt = bundleTransform(MRr, XRr, MUU, XUUinv, MUU, XUU, omega)
    MSt, XSt = bundleTransform(MSr, XSr, MVV, XVVinv, MVV, XVV, omega)
    # returns
    #   1. the real manifold 
    #   2. the graph-style foliation in two directions
    return MRf, XRf, MWf, XWf, MRt, XRt, MUr, XUtr_flat, MSt, XSt, MVr, XVtr_flat
end

# this takes the first coordinate of the input and interprets it as a scalar valued complex map
# divide by z 
# function freqDamp(MS::QPFourierPolynomial{2, 2, fourier_order, min_polyorder, max_polyorder}, XS, r, theta, gamma) where {fourier_order, min_polyorder, max_polyorder}
#     a0id = PolyFindIndex(MS.mexp, [1;0])
#     a0 = XS[1,a0id,fourier_order + 1]
#     #
#     out = zeros(eltype(XS), length(r))
#     for k=1:length(theta)
#         for p=1:size(XS,2), f=-fourier_order:fourier_order
#             out[k] += XS[1,p,fourier_order + 1 + f] * r[k] ^ (sum(MS.mexp[:,p]) - 1) * exp(1im*gamma[k] * (MS.mexp[1,p] - 1 - MS.mexp[2,p]) + 1im*f*theta[k])
#         end
#     end
#     return out, angle.(out), log.(abs.(out)) ./ angle.(out)
# end
