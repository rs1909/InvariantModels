@doc raw"""
    MAPFrequencyDamping(MW::QPFourierPolynomial, XW, MR::QPFourierPolynomial, XR, amp_max; output = Diagonal(I,dim_out))
    
Calculates corrected frequencies and damping ratios for a centre type equilibrium. The inputs are
* `MW`, `XW` is the manifold immersion ``\boldsymbol{W}: \mathbb{R}^2 \times [0,2\pi) \to \mathbb{R}^n``
* `MR`, `XR` is the conjugate dynamics ``\boldsymbol{W}: \mathbb{R}^2 \times [0,2\pi) \to \mathbb{R}^2``
* `amp_max` is the maximum amplitude to be considered
* `output` is a matrix ``\boldsymbol{M}`` that is used to calculate the instantaneous amplitude by pre multiplying the immersion: ``\boldsymbol{M} \boldsymbol{W}``.
The inputs are such that they satisfy the invariance equation
```math
\boldsymbol{W}\left(\boldsymbol{R}\left(\boldsymbol{z},\theta\right),\theta+\omega\right)=\boldsymbol{F}\left(\boldsymbol{W}\left(\boldsymbol{z},\theta\right),\theta\right),
```
where ``\boldsymbol{F}`` is the map of the original dynamics. Given that the invariant manifold is in a normal form, we have the following structure to the conjugate map
```math
\boldsymbol{R}\left(\boldsymbol{z}\right)=\begin{pmatrix}s\left(z,\overline{z}\right)\\
\overline{s}\left(z,\overline{z}\right)
\end{pmatrix}
```
Using a the transformation ``z=\rho\left(r\right)\exp\left(i\beta+i\alpha\circ\rho\left(r\right)\right)`` we can write the manifold immersion as
```math
\widehat{\boldsymbol{W}}\left(r,\beta,\theta\right)=\boldsymbol{W}\left(\rho\left(r\right)\exp\left(i\beta+i\alpha\circ\rho\left(r\right)\right),\rho\left(r\right)\exp\left(-i\beta-i\alpha\circ\rho\left(r\right)\right),\theta\right)
```
so that the invariance equation holds
```math
\widehat{\boldsymbol{W}}\left(R\left(r\right),\beta+T\left(r\right),\theta+\omega\right)=\boldsymbol{F}\left(\widehat{\boldsymbol{W}}\left(r,\beta,\theta\right),\theta\right),
```
where ``R`` and ``T`` are calculated using the same transformation.

The unknown functions ``\rho`` and ``\alpha`` are calculated such that 
1. the amplitude of a tori as a function of ``r`` is the same as ``r`` and 
2. there is zero phase shift between any two tori with different amplitudes with respect to parameter ``\beta``.

Return values are
```
T, R_r, rho, alpha
```
* `T` is the function ``T : [0,\infty) \to \mathbb{R}``
* `R_r` is the function ``r \mapsto R(r) / r``
* `rho` is the same as ``\rho``
* `alpha` is the same as ``\alpha``

The instantaneous frequency is calculated as
```math
\omega(r) = \frac{T(r)}{\Delta t},
```
the instantaneous damping is calculated as
```math
\zeta(r) = -\frac{\log r^{-1} R(r)}{T(r)},
```
where ``\Delta t`` is the sampling period.
"""
function MAPFrequencyDamping(MW::QPFourierPolynomial{dim_out, 2, Wfourier_order, Wmin_polyorder, Wmax_polyorder}, XW, MS::QPFourierPolynomial{2, 2, Sfourier_order, Smin_polyorder, Smax_polyorder}, XS, amp_max; output = Diagonal(I,dim_out)) where {dim_out, Wfourier_order, Wmin_polyorder, Wmax_polyorder, Sfourier_order, Smin_polyorder, Smax_polyorder}
    kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, r_max = ScalingFunctions(MW, XW, amp_max, output = output)
    R, R_r, T = MAPtoPolar(MS, XS, r_max)

    len = maximum(map(m -> length(m.coefficients), (kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, R, R_r, T)))
    S_rho = space(rho)
    That = Fun(r -> T(rho(r)) + gamma(rho(r)) - gamma(R(rho(r))), S_rho)
    Rhat_r = Fun(r -> rho_r(r) * R_r(rho(r)) * kappa_r(R(rho(r))), S_rho)
    return That, Rhat_r, rho, gamma
end

@doc raw"""
    ODEFrequencyDamping(MW::QPFourierPolynomial, XW, MR::QPFourierPolynomial, XR, amp_max; output = Diagonal(I,dim_out))

The inputs and outputs are the same as [`MAPFrequencyDamping`](@ref), except that the interpretation of the input/output is slightly different. The input satisfies the invariance equation
```math
D_1\boldsymbol{W}\left(\boldsymbol{z},\theta\right) \boldsymbol{R}\left(\boldsymbol{z},\theta\right) + D_1\boldsymbol{W}\left(\boldsymbol{z},\theta\right) \omega = \boldsymbol{F}\left(\boldsymbol{W}\left(\boldsymbol{z},\theta\right),\theta\right),
```
where ``\boldsymbol{F}`` is a vector field.

From the output, the instantaneous frequency is 
```math
\omega(r) = T(r),
```
the instantaneous damping is calculated as
```math
\zeta(r) = -\frac{r^{-1} R(r)}{T(r)}.
```
"""
function ODEFrequencyDamping(MW::QPFourierPolynomial{dim_out, 2, Wfourier_order, Wmin_polyorder, Wmax_polyorder}, XW, MS::QPFourierPolynomial{2, 2, Sfourier_order, Smin_polyorder, Smax_polyorder}, XS, amp_max; output = Diagonal(I,dim_out)) where {dim_out, Wfourier_order, Wmin_polyorder, Wmax_polyorder, Sfourier_order, Smin_polyorder, Smax_polyorder}
    kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, r_max = ScalingFunctions(MW, XW, amp_max, output = output)
    R, R_r, T = ODEtoPolar(MS, XS, r_max)

    len = maximum(map(m -> length(m.coefficients), (kappa, kappa_r, D_kappa, rho, rho_r, gamma, R, R_r, T)))
    S_rho = space(rho)
    That = Fun(r -> T(rho(r)) - D_gamma(rho(r))*R(rho(r)), S_rho, 5*len)
    Rhat_r = Fun(r -> R_r(rho(r)) * rho_r(r) * D_kappa(rho(r)), S_rho, 5*len)

    return That, Rhat_r, rho, gamma
end

function MAPtoPolar(MS::QPFourierPolynomial{2, 2, fourier_order, min_polyorder, max_polyorder}, XS, r_max) where {fourier_order, min_polyorder, max_polyorder}
    ords = dropdims(sum(MS.mexp, dims=1), dims=1)
    oid = findall(ords .> 0)
    f_p_r = r -> sum(XS[1,oid,fourier_order+1] .* (r .^ (ords[oid] .- 1)))
    f_p = r -> sum(XS[1,:,fourier_order+1] .* (r .^ ords))
    S_fun = Chebyshev(-0.02*r_max .. 1.02*r_max)
    R = Fun(r -> sign(r) * abs(f_p(r)), S_fun)
    R_r = Fun(r -> abs(f_p_r(r)), S_fun)
    T = Fun(r -> angle(f_p_r(r)), S_fun)
    return R, R_r, T
end

function ODEtoPolar(MS::QPFourierPolynomial{2, 2, fourier_order, min_polyorder, max_polyorder}, XS, r_max) where {fourier_order, min_polyorder, max_polyorder}
    ords = dropdims(sum(MS.mexp, dims=1), dims=1)
    oid = findall(ords .> 0)
    f_p_r = r -> sum(XS[1,oid,fourier_order+1] .* (r .^ (ords[oid] .- 1)))
    f_p = r -> sum(XS[1,:,fourier_order+1] .* (r .^ ords))
    S_fun = Chebyshev(-0.02*r_max .. 1.02*r_max)
    R = Fun(r -> real(f_p(r)), S_fun)
    R_r = Fun(r -> real(f_p_r(r)), S_fun)
    T = Fun(r -> imag(f_p_r(r)), S_fun)
    return R, R_r, T
end

# Inputs
#   MW, XW  -> the manifold immersion
#   amp_max -> the maximum amplitude at which W is valid
# Outputs
#   kappa   -> sqrt (amp square)
#   kappa_r -> kappa / r
#   D_kappa -> D[kappa,r]
#   rho     -> inverse(kappa)
#   rho_r   -> rho/r
#   gamma   -> phase shift 
#   D_gamma -> D[gamma,r]
#   r_max   -> rho(amp_max)
# next up is: calculate kappa
# does need intricate knowledge of QPFourierPolynomial
function ScalingFunctions(MW::QPFourierPolynomial{dim_out, 2, fourier_order, min_polyorder, max_polyorder}, XW, amp_max = 1.0; output = Diagonal(I,dim_out)) where {dim_out, fourier_order, min_polyorder, max_polyorder}
    # we have
    #   W_ipk W_iql r^(p1+p2+q1+q2) d_(p1-p2+q1-q2) d_(k+l)
    # so let l = -k
#     QPFourierPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder)
#            QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field::AbstractNumbers=ℝ)
    @tullio XWres[i,k,l] := output[i,j] * XW[j,k,l]
    Mres = QPPolynomial(1, 1, 0, 0, 2*max_polyorder)
    # alpha is kappa^2 / r^2
    X_alpha = zero(Mres)
    X_g_g = zero(Mres)
    X_g_r = zero(Mres)
    for pid = 1:size(MW.mexp,2), qid = 1:size(MW.mexp,2)
        p1 = MW.mexp[1,pid]
        p2 = MW.mexp[2,pid]
        q1 = MW.mexp[1,qid]
        q2 = MW.mexp[2,qid]
        ord = p1 + p2 + q1 + q2
        fourier_ord = p2 - p1 + (q1 - q2)
        if (fourier_ord == 0) && (ord > 0)
            # here ord is even, because p1 = p2 + q1 - q2
            # hence ord = p1 + p2 + q1 + q2 = 2*p2 + 2*q1
            oidm2 = PolyFindIndex(Mres.mexp, ord-2)
            oidm3 = PolyFindIndex(Mres.mexp, ord-3)
            XW_sq = zero(eltype(XWres))
            for k = -fourier_order:fourier_order
                l = k
                XW_sq += sum(conj(XWres[:,qid,1+fourier_order+l]) .* XWres[:,pid,1+fourier_order+k])
            end
            X_alpha[1,oidm2] += real(XW_sq)
            X_g_g[1,oidm2]   += real((p1-p2) * (q1-q2) * XW_sq)
            if oidm3 != nothing
                X_g_r[1,oidm3] += real((p1+p2) * 1im * (q1-q2) * XW_sq)
            else
                @show real((p1+p2) * 1im * (q1-q2) * XW_sq)
            end
        end
    end
    X_alpha ./= 2*pi
    # now fit Chebyshev polynomials to the required quantities
    # We need to do this because Taylor expansion of polynomials divided is not a good idea
    r_max = InverseScalar(Mres, X_alpha, amp_max, zero(amp_max))
    # kappa
    S_kappa = Chebyshev(-0.02*r_max .. 1.02*r_max)
    kappa   = Fun(a -> a*sqrt(EvalScalar(Mres, X_alpha, a)), S_kappa)
    kappa_r = Fun(a -> sqrt(EvalScalar(Mres, X_alpha, a)), S_kappa)
    D_kappa = Derivative(S_kappa) * kappa
    
    # rho
    k_max = 0.98 * maximum(kappa)
    k_min = 0.98 * minimum(kappa)
    S_rho = Chebyshev(k_min .. k_max)
    rho = Fun(a -> InverseScalar(Mres, X_alpha, a, zero(a)), S_rho)
    rho_r = Fun(a -> InverseScalar(Mres, X_alpha, a, zero(a))/a, S_rho)

    # now phi_deri
    D_gamma = Fun(a -> EvalScalar(Mres, X_g_r, a) / EvalScalar(Mres, X_g_g, a), S_kappa)
    Iop = Integral(S_kappa)
    gamma = Iop * D_gamma
    gamma = gamma - gamma(0.0)
        
    return kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, r_max
end

function EvalScalar(MS::QPPolynomial{1, 1, fourier_order, min_polyorder, max_polyorder}, XS, r::Number) where {fourier_order, min_polyorder, max_polyorder}
    fr = reshape(XS[1,:,fourier_order+1],1,:) * (r .^ dropdims(sum(MS.mexp, dims=1), dims=1))
    return fr[1]
end

function EvalDeriScalar(MS::QPPolynomial{1, 1, fourier_order, min_polyorder, max_polyorder}, XS, r::Number) where {fourier_order, min_polyorder, max_polyorder}
    pws = dropdims(sum(MS.mexp, dims=1), dims=1)
    id = findall(pws .> 0)
    fr = reshape(XS[1,id,fourier_order+1] .* pws[id],1,:) * (r .^ (pws[id] .- 1))
    return fr[1]
end

function InverseScalar(MS::QPPolynomial{1, 1, fourier_order, min_polyorder, max_polyorder}, XS, r::Number, r0::Number; maxit = 100) where {fourier_order, min_polyorder, max_polyorder}
    f  = z_ -> EvalScalar(MS, XS, z_)
    df = z_ -> EvalDeriScalar(MS, XS, z_)
    g  = z_ -> z_ * sqrt(f(z_))
    dg = z_ -> sqrt(f(z_)) + z_ * df(z_) / (2 * sqrt(f(z_)))
    
    xk = r0
    x = r0
    k = 1
    while true
        xk = x
        x = xk - (g(xk) - r) / dg(xk)
        if abs(x-xk) <= 20*eps(typeof(r))
            break
        end
        if k > maxit
            @show abs(x-xk)
            return x
        end
        k += 1
    end
    return x
end

# function CosP_SinQ(p,q)
#     if iseven(p)*iseven(q)
#         return gamma((p+1.0)/2)*gamma((q+1.0)/2) / factorial(div(p+q,2)) / pi
#     else
#         return 0.0
#     end
# end
# 
# function createCircularIntegral(M::QPPolynomial, r)
#     return QPPolynomial(1, 1, 0, 0, PolyOrder(M) + r)
# end
# 
# # Calculate the integral the g(r) = 1/(2pi) \int_0^{2\pi} f(r Cos(t), r Sin(t) ) * cos(t)^c * sin(t)^s dt
# function circularIntegrate(MO, M::QPPolynomial, X, c_, s_, r_)
#     XO = zero(MO)
#     XI = vec(X)
#     for k=1:size(M.mexp,2)
#         p = M.mexp[1,k]
#         q = M.mexp[2,k]
#         id = findfirst(isequal(p+q+r_), MO.mexp[1,:])
#         if id != nothing
#             XO[1,id,1] += CosP_SinQ(p+c_,q+s_) * XI[k]
#         elseif abs(CosP_SinQ(p+c_,q+s_) * XI[k]) >= eps(XI[k])
#             println("NOT FOUND p+c=", p+c_, " q+s=", q+s_, " p+q+r=", p+q+r_, " CosP_SinQ(p+c,q+s) * XI[k]=", CosP_SinQ(p+c_,q+s_) * XI[k])
#         end
#     end
#     return XO
# end
# 
# function ScalingFunctions(MW::QPPolynomial{dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field}, XW, amp_max; output = Diagonal(I,dim_out)) where {dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field}
#     # make sure that it is at the origin, as we are measuring from the qp-orbit
#     setConstantPart!(MW, XW, zero(size(XW,1)))
#     MWt = QPPolynomial(size(output,1), dim_in, fourier_order, min_polyorder, max_polyorder, field)
#     @tullio XWt[i,k,l] := output[i,j] * XW[j,k,l]
#     #---------------------------------------------------------
#     #
#     # Calculate the geometry on the manifold
#     #
#     #---------------------------------------------------------
#     # In 2 dims
#     #   z = [ r Cos(t), r Sin(t) ]
#     #   D1 W(r, t) = DW( T(r,t) ) [ Cos(t), Sin(t) ]
#     #   D2 W(r, t) = DW( T(r,t) ) [ -r Sin(t), r Cos(t) ]
#     # We need
#     #   A00 = Int < D2 W(r,t),    W(r,t) > dt / (2 Pi) = r Int W^T . W dt
#     #   A12 = Int < D1 W(r,t), D2 W(r,t) > dt / (2 Pi) = r Int [ Cos(t), Sin(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] dt
#     #   A22 = Int < D2 W(r,t), D2 W(r,t) > dt / (2 Pi) = r^2 Int [ -Sin(t), Cos(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] dt
#     # Let 
#     #   DW^T * DW = ( a11 a12 )
#     #               ( a21 a22 )
#     #   with r^(p+q) Cos^p * Sin^q coefficient, i.e., -> (p,q)
#     # Then
#     # 1. A12
#     #   [ Cos(t), Sin(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] = 
#     #   = - a11 r Cos[t] Sin[t] + a12 r Cos[t]^2 - a21 r Sin[t]^2 + a22 r Cos[t] Sin[t]
#     #   = r^(p+q+1) [ (a22 - a11) Cos^(p+1) * Sin^(q+1)
#     #              + a12 * Cos^(p+2) * Sin^(q)
#     #              - a21 * Cos^(p) * Sin^(q+2) ]
#     # 2. A22
#     #   [ -Sin(t), Cos(t) ] . DW^T * DW . [ -Sin(t), Cos(t) ] = 
#     #   = a11 r^2 Sin[t]^2 - a12 r^2 Cos[t] Sin[t] - a21 r^2 Cos[t] Sin[t] + a22 r^2 Cos[t]^2 
#     #   = r^(p+q+2) [ -(a12 + a21) Cos^(p+1) * Sin^(q+1)
#     #              + a22 * Cos^(p+2) * Sin^(q)
#     #              + a11 * Cos^(p) * Sin^(q+2) ]
#     # For the integrals, we need to use the Gamma special function an factorial
#     #   Int Cos^{2n}(t) Sin^{2m}(t) dt = Gamma(n+1/2) Gamma(m+1/2) / pi / (m+n)!
#     # Solve
#     #   delta' = - A12/A22
#     #   kappa' = A10 + A20 * delta'
#     #---------------------------------------------------------
#     din = size(XWt,1)
#     dout = size(MWt.mexp,1)
#     order = PolyOrder(MWt)
# 
#     # for A00
#     M_WtWt = QPPolynomial(dout, 1, 0, 0, 2*order) # DensePolyManifold(dout, 1, 2*order)
#     X_WtWt = zero(M_WtWt)
#     DensePolySquared!(M_WtWt, X_WtWt, MWt, XWt)
#     # DW^T . W
#     M_DWtr_W = QPPolynomial(dout, dout, 0, 0, 2*order-1) # DensePolyManifold(dout, dout, 2*order-1)
#     X_DWtr_W = zero(M_DWtr_W)
#     DensePolyDeriTransposeMul!(M_DWtr_W, X_DWtr_W, MWt, XWt)
# 
#     # DW^T . DW
#     M_DWtr_DW = QPPolynomial(dout, dout, 0, 0, 2*order-1) # DensePolyManifold(dout, dout, 2*order-1)
#     X_DWtr_DW = zeroJacobianSquared(M_DWtr_DW)
#     DensePolyJabobianSquared!(M_DWtr_DW, X_DWtr_DW, MWt, XWt)
#     
#     # 2 is sufficient, 3 is important, of we want to integrate kappa accurately
#     MO = createCircularIntegral(M_DWtr_W, 3)
#     X_00    = circularIntegrate(MO, M_WtWt,     X_WtWt[1,:], 0, 0, 0 - 2) # division by r^2
#     X_12_11 = circularIntegrate(MO, M_DWtr_DW, -X_DWtr_DW[1,1,:], 1, 1, 1 - 2) # -1 is division by r
#     X_12_12 = circularIntegrate(MO, M_DWtr_DW,  X_DWtr_DW[1,2,:], 2, 0, 1 - 2)
#     X_12_21 = circularIntegrate(MO, M_DWtr_DW, -X_DWtr_DW[2,1,:], 0, 2, 1 - 2)
#     X_12_22 = circularIntegrate(MO, M_DWtr_DW,  X_DWtr_DW[2,2,:], 1, 1, 1 - 2)
#     X_12 = X_12_11 + X_12_12 + X_12_21 + X_12_22
#         
#     X_22_11 = circularIntegrate(MO, M_DWtr_DW,  X_DWtr_DW[1,1,:], 0, 2, 2 - 2)
#     X_22_12 = circularIntegrate(MO, M_DWtr_DW, -X_DWtr_DW[1,2,:], 1, 1, 2 - 2)
#     X_22_21 = circularIntegrate(MO, M_DWtr_DW, -X_DWtr_DW[2,1,:], 1, 1, 2 - 2)
#     X_22_22 = circularIntegrate(MO, M_DWtr_DW,  X_DWtr_DW[2,2,:], 2, 0, 2 - 2)
#     X_22 = X_22_11 + X_22_12 + X_22_21 + X_22_22
#         
#     # X_00 is kappa_r squared
#     # - X_12 / X_22 is gamma_derivative
#     A00 = t -> Eval(MO, X_00, [0], [t])[1]
#     A12 = t -> Eval(MO, X_12, [0], [t])[1]
#     A22 = t -> Eval(MO, X_22, [0], [t])[1]
# 
#     r_max = InverseScalar(MO, X_00, amp_max, zero(amp_max))
#     # now fit Chebyshev polynomials to the required quantities
#     # We need to do this because Taylor expansion of polynomials divided is not a good idea
#     # kappa
#     S_kappa = Chebyshev(-0.02*r_max .. 1.02*r_max)
#     kappa   = Fun(a -> a*sqrt(A00(a)), S_kappa)
#     kappa_r = Fun(a -> sqrt(A00(a)), S_kappa)
#     D_kappa = Derivative(S_kappa) * kappa
#     
#     # rho
#     k_max = 0.98 * maximum(kappa)
#     k_min = 0.98 * minimum(kappa)
#     S_rho = Chebyshev(k_min .. k_max)
#     rho = Fun(a -> InverseScalar(MO, X_00, a, zero(a)), S_rho)
#     rho_r = Fun(a -> InverseScalar(MO, X_00, a, zero(a))/a, S_rho)
# 
#     # now phi_deri
#     D_gamma = Fun(a -> -A12(a)/A22(a), S_kappa)
#     Iop = Integral(S_kappa)
#     gamma = Iop * D_gamma
#     gamma = gamma - gamma(0.0)
#     
#     @show X_00
#     @show X_12
#     @show X_22
#     return kappa, kappa_r, D_kappa, rho, rho_r, gamma, D_gamma, r_max
# end
