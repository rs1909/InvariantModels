#------------------------------------------------------------------------------------------------------------------------------------------
#
# Putting the manifold back into real coordinates
#
#------------------------------------------------------------------------------------------------------------------------------------------
    # function ComplexReal2 comes from "directfoliation.jl"
    # MSf, XSf -> S
    # MWf, XWf -> manifold decomposition
    # MWd, XWd -> linear bundle decomposition
    # output:
    # W = Wd . Wf . R^-1
    # S = R . S . R^-1
    function ManifoldToReal(MSf, XSf, MWf, XWf, MWd, XWd)
        MWFd, XWFd = toFourier(MWd, XWd)
        XWd_Wf = zero(MWf)
        QPFourierPolynomialSubstitute!(MWf, XWd_Wf, MWFd, XWFd, MWf, XWf)
        M2, X2R, X2C = ComplexReal2(fourier_order)
        
        XWd_Wf_C = zero(MWf)
        QPFourierPolynomialSubstitute!(MWf, XWd_Wf_C, MWf, XWd_Wf, M2, X2C)
        # S ->
        XRoS = zero(MSf)
        QPFourierPolynomialSubstitute!(MSf, XRoS, M2, X2R, MSf, XSf)
        XSt = zero(MSf)
        QPFourierPolynomialSubstitute!(MSf, XSt, MSf, XRoS, M2, X2C)

        MWr, XWr = fromFourier(MWf, XWd_Wf_C)
        MSr, XSr = fromFourier(MSf, XSt)
        return MSr, XSr, MWr, XWr
    end

#------------------------------------------------------------------------------------------------------------------------------------------
#
# direct MANIFOLD from a map
#
#------------------------------------------------------------------------------------------------------------------------------------------

# input: 
#   M, X, omega represent a system that is transformed into a diagonal form.
#       It is assumed that 
#           - the constant part is zero
#           - the linear part of the system diagonal and autonomous (reducible system)
#   sel_ is the set of eigenvalues we need
#   resonance: 
#       true of internal resonances with non-zero frequencies are allowed
#       false (default) if we want the output be completely autonomous (even if it blows up)
# output:
#   MS, XS, MW, XW
#       MS, XS are the reduced dynamics
#       MW, XW are the submersion
function iQPManifold(M::QPFourierPolynomial{dim_out, dim_in, fourier_order, min_polyorder, max_polyorder}, X, omega, sel_; resonance::Bool = true, threshold = 0.1, rom_order = max_polyorder) where {dim_out, dim_in, fourier_order, min_polyorder, max_polyorder}
    sel = vec(sel_)
    # co-dimension of the foliation
    f_dim = length(sel)
    # create the output
    MW = QPFourierPolynomial(dim_in, f_dim, fourier_order, min_polyorder, max_polyorder)
    XW = zero(MW)
    # create ROM
    MS = QPFourierPolynomial(f_dim, f_dim, fourier_order, min_polyorder, max_polyorder)
    XS = zero(MS)
    # We are solving the invariance equation
    #   W(t+omega, S(t, x)) = F(t, W(t, x))
    # it is assumed that linear part of F is diagonal and only has a constant part
    LP = GetLinearPart(M, X)
    # Lambda is an averaged linear part
    Lambda = diag(LP[:,:,fourier_order+1])
    reva = log.(abs.(Lambda))
    println("iQPManifold: R. SPECTRAL QUOTIENT = ", minimum(reva[sel]) / maximum(reva))
    d_sel = setdiff(1:dim_out,sel)
    if !isempty(d_sel)
        println("iQPManifold: S. SPECTRAL QUOTIENT = ", minimum(reva[d_sel]) / maximum(reva))
    end
    # clearing the time-dependent parts of M, X
    LP .= zero(eltype(LP))
    LP[:,:,fourier_order+1] .= Diagonal(Lambda)
    SetLinearPart!(M, X, LP)
    # Clearing the constant part
    SetConstantPart!(M, X, zero(eltype(X)))
    # setting the linear part
    LW = GetLinearPart(MW, XW)
#     @show size(LU[:,sel,fourier_order+1])
    LW[sel,:,fourier_order+1] .= one(LW[sel,:,fourier_order+1])
    SetLinearPart!(MW, XW, LW)
    #
    LS = GetLinearPart(MS, XS)
    LS[:,:,fourier_order+1] .= Diagonal(Lambda[sel])
    SetLinearPart!(MS, XS, LS)
    # the residual, after subs R(t,x) = W(t+omega, S(t, x)) - F(t, W(t, x))
    # same polynomial as MW
    XR = zero(MW)
    # the Fourier representetaion starts with Exp(i * (- fourier_order) * t) as per convention
    # so shifting U(t+omega,.) is the same as multiplying with Exp(i * k * omega) ... k = - fourier_order : fourier_order
    
    # create the residual
    #   R(t,x) = W(t+omega, S(t, x)) - F(t, W(t, x))
    X_WoS = zero(MW)
    tabs_WoS = QPFourierSubstituteTabs(MW, X_WoS, MW, XW, MS, XS)
    QPFourierPolynomialSubstitute!(MW, X_WoS, MW, XW, MS, XS, tabs_WoS..., shift = omega)
    X_FoW = zero(MW)
    tabs_FoW = QPFourierSubstituteTabs(MW, X_FoW, M, X, MW, XW)
    QPFourierPolynomialSubstitute!(MW, X_FoW, M, X, MW, XW, tabs_FoW...)
    XR .= X_FoW .- X_WoS
#     mx = findall(abs.(GetLinearPart(MU, X_SoU)) .> 1e-6)

    for ord = 2:PolyOrder(MW)
        id = PolyOrderIndices(MW, ord)
        # within this, check each term for each dimension and Fourier order
        for ij in id
            Sij = PolyFindIndex(MS.mexp, MW.mexp[:,ij])
            for i0=1:size(XW,1), kp=1:size(XW,3)
                k = kp - 1 - fourier_order
                den = prod(Lambda[sel] .^ MW.mexp[:,ij]) * exp(1im*k*omega) - Lambda[i0]
                denAngle = sum(angle.(Lambda[sel]) .* MW.mexp[:,ij]) + k*omega - angle(Lambda[i0])
                if (abs(denAngle) > threshold) || (ord > rom_order)
                    XW[i0, ij, kp] = XR[i0, ij, kp] / den
                    # XS -> no need
                else
                    if (i0 in sel) && ((k == 0) || resonance)
                        # internal resonance
                        Si0 = findfirst(isequal(i0), sel)
                        XS[Si0, Sij, kp] = XR[i0, ij, kp]
                        println("Internal resonance: ", @sprintf("abs=%.5e", abs(den)), " i0=", i0, " ij=", MW.mexp[:,ij], " k=", k, " Sij=", MS.mexp[:,Sij], @sprintf(" angle=%.5e", abs(denAngle)))
                    else
                        # unmitigated resonance
                        println("Unmitigated resonance: ", @sprintf("abs=%.5e", abs(den)), " i0=", i0, " ij=", MW.mexp[:,ij], " k=", k)
                        XW[i0, ij, kp] = XR[i0, ij, kp] / den
                    end
                end
            end
        end
        QPFourierPolynomialSubstitute!(MW, X_WoS, MW, XW, MS, XS, tabs_WoS..., shift = omega)
        QPFourierPolynomialSubstitute!(MW, X_FoW, M, X, MW, XW, tabs_FoW...)
        XR .= X_FoW .- X_WoS
    end
#     @show abs.(XR)
    @show maximum(abs.(XR))
    return MS, XS, MW, XW
end

#------------------------------------------------------------------------------------------------------------------------------------------
#
# direct MANIFOLD from a map
#
#------------------------------------------------------------------------------------------------------------------------------------------

# input: 
#   M, X, omega represent a system that is transformed into a diagonal form.
#       It is assumed that 
#           - the constant part is zero
#           - the linear part of the system diagonal and autonomous (reducible system)
#   sel_ is the set of eigenvalues we need
#   resonance: 
#       true of internal resonances with non-zero frequencies are allowed
#       false (default) if we want the output be completely autonomous (even if it blows up)
# output:
#   MS, XS, MW, XW
#       MS, XS are the reduced dynamics
#       MW, XW are the submersion
function iQPODEManifold(M::QPFourierPolynomial{dim_out, dim_in, fourier_order, min_polyorder, max_polyorder}, X, omega, sel_; resonance::Bool = true, threshold = 0.1, rom_order = max_polyorder) where {dim_out, dim_in, fourier_order, min_polyorder, max_polyorder}
    @assert min_polyorder == 0 "Wrong input polynomial order."
    sel = vec(sel_)
    # co-dimension of the foliation
    f_dim = length(sel)
    # create the output
    MW = QPFourierPolynomial(dim_in, f_dim, fourier_order, min_polyorder, max_polyorder)
    XW = zero(MW)
    # create ROM
    MS = QPFourierPolynomial(f_dim, f_dim, fourier_order, min_polyorder, max_polyorder)
    XS = zero(MS)
    # We are solving the invariance equation
    #   D1 W(z, t) S(z,t) + D2 W(z, t) omega = F(W(z, t), t)
    # it is assumed that linear part of F is diagonal and only has a constant part
    LP = GetLinearPart(M, X)
    # Lambda is an averaged linear part
    Lambda = diag(LP[:,:,fourier_order+1])
    reva = real.(Lambda)
    println("iQPODEManifold: R. SPECTRAL QUOTIENT = ", minimum(reva[sel]) / maximum(reva))
    println("iQPODEManifold: S. SPECTRAL QUOTIENT = ", minimum(reva[setdiff(1:dim_out,sel)]) / maximum(reva))
    # clearing the time-dependent parts of M, X
    LP .= zero(eltype(LP))
    LP[:,:,fourier_order+1] .= Diagonal(Lambda)
    SetLinearPart!(M, X, LP)
    # Clearing the constant part
    SetConstantPart!(M, X, zero(eltype(X)))
    # setting the linear part
    LW = GetLinearPart(MW, XW)
#     @show size(LU[:,sel,fourier_order+1])
    LW[sel,:,fourier_order+1] .= one(LW[sel,:,fourier_order+1])
    SetLinearPart!(MW, XW, LW)
    #
    LS = GetLinearPart(MS, XS)
    LS[:,:,fourier_order+1] .= Diagonal(Lambda[sel])
    SetLinearPart!(MS, XS, LS)
    # the residual, after subs R(x,t) = F(W(z, t),t) - D1 W(z,t) S(z,t) - D2 W(z,t) omega
    # same polynomial as MW
    XR = zero(MW)
    # the Fourier representetaion starts with Exp(i * (- fourier_order) * t) as per convention
    # so shifting U(t+omega,.) is the same as multiplying with Exp(i * k * omega) ... k = - fourier_order : fourier_order
    
    # create the residual
    #   R(t,x) = D1 W(z,t) S(z,t) + D2 W(z,t) omega - F(W(z,t),t)
    X_D1W_S = zero(MW)
    # maybe create the mul table???
    QPFourierDeriMul!(MW, X_D1W_S, MW, XW, MS, XS)
    X_D2W_omega = thetaDerivative(MW, XW, omega)
    X_FoW = zero(MW)
    tabs_FoW = QPFourierSubstituteTabs(MW, X_FoW, M, X, MW, XW)
    QPFourierPolynomialSubstitute!(MW, X_FoW, M, X, MW, XW, tabs_FoW...)
    XR .= X_FoW .- X_D1W_S .- X_D2W_omega

    for ord = 2:PolyOrder(MW)
        id = PolyOrderIndices(MW, ord)
        # within this, check each term for each dimension and Fourier order
        for ij in id
            Sij = PolyFindIndex(MS.mexp, MW.mexp[:,ij])
            for i0=1:size(XW,1), kp=1:size(XW,3)
                k = kp - 1 - fourier_order
                den = sum(Lambda[sel] .* MW.mexp[:,ij]) + 1im*k*omega - Lambda[i0]
                if (abs(imag(den)) > threshold) || (ord > rom_order)
                    XW[i0, ij, kp] = XR[i0, ij, kp] / den
                    # XS -> no need
                else
                    if (i0 in sel) && ((k == 0) || resonance)
                        # internal resonance
                        Si0 = findfirst(isequal(i0), sel)
                        XS[Si0, Sij, kp] = XR[i0, ij, kp]
                        println("Internal resonance: ", abs(den), " i0=", i0, " ij=", MW.mexp[:,ij], " k=", k, " Sij=", MS.mexp[:,Sij], @sprintf(" angle=%.5e", abs(imag(den))))
                    else
                        # unmitigated resonance
                        println("Unmitigated resonance: ", abs(den), " i0=", i0, " ij=", MW.mexp[:,ij], " k=", k)
                        XW[i0, ij, kp] = XR[i0, ij, kp] / den
                    end
                end
            end
        end
        QPFourierDeriMul!(MW, X_D1W_S, MW, XW, MS, XS)
        X_D2W_omega = thetaDerivative(MW, XW, omega)
        QPFourierPolynomialSubstitute!(MW, X_FoW, M, X, MW, XW, tabs_FoW...)
        XR .= X_FoW .- X_D1W_S .- X_D2W_omega
    end
#     @show abs.(XR)
    @show maximum(abs.(XR))
    return MS, XS, MW, XW
end

@doc raw"""
    QPMAPTorusManifold(MP::QPPolynomial, XP, omega, SEL; 
                            threshold = 0.1, 
                            resonance = false, 
                            rom_order = max_polyorder)

The inputs are the following
* `MP`, `XP` the Poincare map that describes the dynamics.
* `omega` the phase shift at each time-step
* `SEL` which spectral bundles to use for the calculation of the invariant manifold
* `threshold` if a near resonance is greater than this value, it is ignored
* `resonance`: when `false` the conjugate map is reduced to an autonomous system, even if there are near resonances. 
    Otherwise all near resonances are accounted for in the conjugate map.
* `rom_order` what is the highest order for which near resonances are taken into account. It defaults to the order of the Poincare map `MP`, `XP`.

This function does three calculations
1. Calculates the invariant torus using Newton's method. The startnig iteration is at the origin.
2. Calculates the linear system about the invariant torues and decomposes it into invariant vector bundles. The bundles are ordered by the corresponding spectrum. The bundles with the greatest spectral radius appear first.
3. Given the invariant bundles about the torus, calculate the invariant manifold tangent to the selected vector bundles. The selection is done by `SEL`.

Return values are:
```
MK, XK, MSn, XSn, MWn, XWn, MSd, XSd, XWre
```
* `MK`, `XK` represent the invariant torus
* `MSn`, `XSn` the normal form of the conjugate dynamics on the invariant manifold
* `MWn`, `XWn` the manifold immersion in the coordinate system of the invariant vector bundle, selected by `SEL`
* `MSd`, `XSd` the full system `MP`, `XP` transformed to the coordinate system of the vector bundle decomposition
* `XWre` the transformation from the vector bundles to the original coordinate system
"""
function QPMAPTorusManifold(MP::QPPolynomial{dim, dim, fourier_order, min_polyorder, max_polyorder, ℝ}, XP, omega, SEL; 
                            threshold = 0.1, 
                            resonance = false, 
                            rom_order = max_polyorder) where {dim, fourier_order, min_polyorder, max_polyorder}
    MK = QPConstant(dim, fourier_order)
    XK = zero(MK)
    torusId!(MK, XK, MP, XP, omega)

    canMap = (x,t) -> Eval(MP, XP, t, Eval(MK, XK, t) + x) - Eval(MK, XK, t + omega)
    XP0 = fromFunction(MP, canMap)

    # the linear transformation
    Lambda, MWd, XWd, MUd, XUd, MUFW, XUFW, XWre = smoothestDecomposition(MP, XP0, omega)
    # the Manifold transformation
    # MFWd, XFWd = toFourier(MWd, XWd)
    MSd, XSd = toFourier(MUFW, XUFW)
    MSn, XSn, MWn, XWn = iQPManifold(MSd, XSd, omega, SEL, resonance = resonance, threshold = threshold, rom_order = rom_order)
    MWdf, XWdf = toFourier(MWd, XWd)
    XWnf = zero(MWn)
    QPFourierPolynomialSubstitute!(MWn, XWnf, MWdf, XWdf, MWn, XWn)
    return MK, XK, MSn, XSn, MWn, XWnf, MSd, XSd, XWre
end

@doc raw"""
    QPODETorusManifold(MP::QPPolynomial, XP, omega, SEL; 
                            threshold = 0.1, 
                            resonance = false, 
                            rom_order = max_polyorder)

This is the equivalent version of [`QPMAPTorusManifold`](@ref) for vector fields.
    
The inputs are the following
* `MP`, `XP` the vector field that describes the dynamics.
* `omega` is the forcing frequency
* `SEL` which spectral bundles to use for the calculation of the invariant manifold
* `threshold` if a near resonance is greater than this value, it is ignored
* `resonance`: when `false` the conjugate map is reduced to an autonomous system, even if there are near resonances. 
    Otherwise all near resonances are accounted for in the conjugate map.
* `rom_order` what is the highest order for which near resonances are taken into account. It defaults to the order of the Poincare map `MP`, `XP`.

This function does three calculations
1. Calculates the invariant torus using Newton's method. The startnig iteration is at the origin.
2. Calculates the linear system about the invariant torues and decomposes it into invariant vector bundles. The bundles are ordered by the corresponding spectrum. The bundles with the greatest spectral radius appear first.
3. Given the invariant bundles about the torus, calculate the invariant manifold tangent to the selected vector bundles. The selection is done by `SEL`.

Return values are:
```
MK, XK, MSn, XSn, MWn, XWn, MSd, XSd, XWre, Lambda
```
* `MK`, `XK` represent the invariant torus
* `MSn`, `XSn` the normal form of the conjugate dynamics on the invariant manifold
* `MWn`, `XWn` the manifold immersion in the coordinate system of the invariant vector bundle, selected by `SEL`
* `MSd`, `XSd` the full system `MP`, `XP` transformed to the coordinate system of the vector bundle decomposition
* `XWre` the transformation from the vector bundles to the original coordinate system
* `Lambda` is the linear part of the system about the invariant torus in the coordinate system of the vector bundles. This is for debug purposes only, it should be a diagonal matrix for each collocation point.
"""
function QPODETorusManifold(MP::QPPolynomial{dim, dim, fourier_order, min_polyorder, max_polyorder, ℝ}, XP, omega_ode, SEL; 
                            threshold = 0.1, 
                            resonance = false, 
                            rom_order = max_polyorder) where {dim, fourier_order, min_polyorder, max_polyorder}
    MK = QPConstant(dim, fourier_order)
    XK = zero(MK)
    ODEtorusId!(MK, XK, MP, XP, omega_ode) # in qptorusid.jl

    XDK = thetaDerivative(MK, XK, omega_ode)

    # moving the torus to the origin
    canMap = (x,t) -> Eval(MP, XP, t, Eval(MK, XK, t) + x) - Eval(MK, XDK, t)
    XP0 = fromFunction(MP, canMap)

    # creating the vector bundles about the origin
    Lambda, MWd, XWd, MUd, XUd, MUFW, XUFW, XWre = smoothestDecomposition(MP, XP0, omega_ode, ODE = true)
    MSd, XSd = toFourier(MUFW, XUFW)
    MSn, XSn, MWn, XWn = iQPODEManifold(MSd, XSd, omega_ode, SEL, resonance = resonance, threshold = threshold, rom_order = rom_order)
    MWdf, XWdf = toFourier(MWd, XWd)
    XWnf = zero(MWn)
    QPFourierPolynomialSubstitute!(MWn, XWnf, MWdf, XWdf, MWn, XWn)
    return MK, XK, MSn, XSn, MWn, XWnf, MSd, XSd, MWd, XWd, XWre, Lambda
end

# calculates the residual of the manifold
# data is in polar coordinates and complex conjugate
# these are in Fourier form
function ODEManifoldAccuracy(MP::QPPolynomial, XP, MK, XK, MW::QPFourierPolynomial, XW, MS::QPFourierPolynomial, XS, omega_ode, thetaZ, dataZ)
    thetaWZ = FourierWeights(MW, thetaZ)
#     @show typeof(thetaWZ), typeof(dataZ), typeof(MW)
    WoZ = Eval(MW, XW, thetaWZ, dataZ)
    KoZ = Eval(MK, XK, thetaZ)
    FoWoZ = Eval(MP, XP, thetaZ, real.(WoZ .+ KoZ))
    SoZ = Eval(MS, XS, thetaWZ, dataZ)
    D1W_SoZ = JF_dx(MW, XW, thetaWZ, dataZ, SoZ)
    XDK = thetaDerivative(MK, XK, omega_ode)
    X_D2W_omega = thetaDerivative(MW, XW, omega_ode)
#     @show Eval(MW, X_D2W_omega, thetaWZ, dataZ)
    res = FoWoZ - D1W_SoZ - Eval(MW, X_D2W_omega, thetaWZ, dataZ) - Eval(MK, XDK, thetaZ)

    return res, WoZ
end

function MAPManifoldAccuracy(MP::QPPolynomial, XP, MK, XK, MW::QPFourierPolynomial, XW, MS::QPFourierPolynomial, XS, omega, thetaZ, dataZ)
    thetaWZ = FourierWeights(MW, thetaZ)
    WoZ = Eval(MW, XW, thetaWZ, dataZ)
    KoZ = Eval(MK, XK, thetaZ)
    FoWoZ = Eval(MP, XP, thetaZ, real.(WoZ .+ KoZ))
    SoZ = Eval(MS, XS, thetaWZ, dataZ)
    thetaWZD = FourierWeights(MW, thetaZ .+ omega)
    WoSoZ = Eval(MW, XW, thetaWZD, SoZ)
    KoZD = Eval(MK, XK, thetaZ .+ omega)
    #
    res = FoWoZ - WoSoZ - KoZD

    return res, WoZ
end

function ErrorStatistics(OnManifoldAmplitude, resU, maxAmp, bins = (40,20))
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
    
    return hsU, errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX
end
