    function findEpsilonNearest!(found, rest, d, epsilon)
        l0 = length(found)
        push!(found, unique([rest[x.I[1]] for x in findall( d[rest,found] .< epsilon)])...)
        unique!(found)
        setdiff!(rest, found)
        l1 = length(found)
        if l1 - l0 != 0
            return findEpsilonNearest!(found, rest, d, epsilon)
        end
        return found, rest, l1 - l0
    end
    
    # d is the distance matrix between elements of the eigenvalues
    function findEpsilonClusters(d, epsilon)
        found = []
        rest = collect(1:size(d,1))
        clusters = []
        while ~isempty(rest)
            found = [rest[1]]
            deleteat!(rest,1)
            f0, r0 = findEpsilonNearest!(found, rest, d, epsilon)
            push!(clusters, f0)
            rest = r0
        end
        return clusters
    end

function orthognaliseLeftBundle(XU)
    XUr = zero(XU)
    for k=1:size(XU,3)
        F = svd(transpose(XU[:,:,k]))
        XUr[:,:,k] .= transpose(F.U * F.Vt)
    end
    return XUr
end
    
# assumes oscillatory dynamics in each cluster
# 1. test substitutions
# 2. figure out the exact form of the decomposition, given that we calculate the eigenvectors of the transfer operator
# 3. check that it actually decomposes into a constant matrix
# 4. checks if there is a delta sized gap between annuli of eigenvalues, then separates that into categories
# XU, XW, XUre, XWre, XUreOrth, XWreOrth, Lambda = vectorBundles(fourier_order, B, omega; orthogonal = false, sel = 1:size(B,2), ODE = false, Tstep = 1.0)

function vectorBundles(fourier_order, B, omega; orthogonal = false, sel = 1:size(B,2), ODE = false, Tstep = 1.0)
    SEL = vec(sel)
    dim_in = size(B,2)
    grid = range(0,2*pi,length=2*fourier_order+2)[1:end-1]
#     @show size(B)
    if ODE
        T = differentialMinusDiagonalOperator(grid, B, omega)
    else
        T = transferOperator(grid, permutedims(B,(1,2,3)), omega) # corrected -omega
    end
    eva, eve = eigen(T)
    # right bundles 1st: n - dimension 2nd: collocation point, 3rd: index of eigenvector
    eve_reshape = reshape(eve, dim_in, :, size(eve,2))
    # left bundles -> set of row vectors. 1st index is the index of the eigenvector
    left_eve_reshape = reshape(inv(eve), size(eve,1), dim_in, :)
    
    # kmeans clustering of eigenvalues about concentric circles
    # each cluster must have integer multiples of (2 ell + 1) elements
    if ODE
        aeva = reshape(real.(eva), 1, :)
    else
        aeva = reshape(abs.(eva), 1, :)
    end
    @show aeva
    ncl = dim_in
    local R
    while true
        R = kmeans(aeva, ncl)
        if all(mod.(R.counts, 2*fourier_order+1) .== 0) && (length(R.centers) == length(unique(R.centers)))
            break
        elseif ncl > div(dim_in,2)
            ncl -= 1
        else
            println("Cannot find clusters.")
            return nothing
        end
    end
    @show R.centers
    @show R.assignments
    @show clusters = [findall(R.assignments .== k) for k in sortperm(vec(R.centers[1:maximum(R.assignments)]), rev=true)]
    # clustering per magnitude of eigenvalues
#     dist = [abs(a-b) for a in abs.(eva), b in abs.(eva)]
#     clusters = findEpsilonClusters(dist, epsilon)
    # start with the longest cluster
#     sort!(clusters, lt = (x,y) -> length(x) > length(y))
#     @assert length(clusters) <= dim_in "Too many clusters = $(length(clusters))."
    
    # calculating roughness
    FM = FourierMatrix(grid)
    @tullio eve_fourier[p,l,q] := FM[l,k] * eve_reshape[p,k,q]
    # find equivalence classes of eigenvectors
    # calculate the amplitude of each Fourier coeffient, for each eigenvector: 1: Fourier amplitude 2: index of eigenvector
    eve_fourier_amplitude = sqrt.(dropdims(sum(real.(eve_fourier .* conj(eve_fourier)), dims=1),dims=1))
    # normalise for the maximum value
    eve_fourier_amplitude ./= sqrt.(sum(eve_fourier_amplitude .^ 2, dims=1))
    # finding the most central ones
    # maybe better just to find one with the highest coefficient at the lowest frequency?
    # or the one closest to the forcing frequency?
    scales = [2^(abs(k)) for k in -fourier_order:fourier_order]
    roughness = transpose(eve_fourier_amplitude) * scales
    for cl in clusters
        cl_rough = sortperm(roughness[cl])
        println(" E = ", eva[cl[cl_rough[1]]])
    end

    # select the least rough vectors and create the transformation
    c_sel = []
    i_sel = []
    r_sel = []
    for cl in clusters
        @show size(cl)
        if mod(length(cl), 2*fourier_order+1) != 0
            println("Wrong number of eigenvalues in a cluster.")
            return nothing
        end
        cl_dim = div(length(cl), 2*fourier_order+1)
#         display(abs.(eva[cl]))
        cl_rough = sortperm(roughness[cl])
        if cl_dim == 1
            # one dimensional bundle, find the real eigenvalue
            m, id = findmin(abs.(imag.(eva[cl])))
            push!(c_sel, cl[id])
            push!(r_sel, cl[id])
        else
            m1, id1 = findmin(roughness[cl])
            m2, id2 = findmin(abs.(eva[cl] .- conj(eva[cl[id1]])))
            @show norm(eve_reshape[:,:,cl[id1]] - conj.(eve_reshape[:,:,cl[id2]]))
            # prefer positive imaginary first
            if imag(eva[cl[id1]]) < 0
                a = id1
                id1 = id2
                id2 = a
            end
            push!(c_sel, cl[id1], cl[id2])
            push!(i_sel, cl[id1])
        end
        if length(c_sel) >= dim_in
            break
        end
    end
    #
    if length(c_sel) != dim_in 
        println("Dimension mismatch. Selected #eigenvalues = $(length(c_sel)) vs. system dimension $(dim_in).")
        for cl in clusters
            sr = sortperm(abs.(eva[cl]))
            println("eigenvalues", mapreduce(x -> @sprintf(" %.5e", x), *, abs.(eva[cl[sr]])))
            println("angles     ", mapreduce(x -> @sprintf(" %.5e", x), *, angle.(eva[cl[sr]])/pi))
        end
        @assert false "You have seen the error..."
    end
    Lambda = Diagonal(eva[c_sel])
    println("Eigenvalues")
#     display(eva[c_sel])
    csel = setdiff(1:dim_in, SEL)
    if ODE
        println("Damping")
        display(real.(eva[c_sel]) ./ abs.(imag.(eva[c_sel])))
        println("Frequency")
        display(imag.(eva[c_sel]))
        reva = real.(eva[c_sel])
        println("R. SPECTRAL QUOTIENT = ", minimum(reva[SEL]) / maximum(reva))
        if !isempty(csel) 
            println("S. SPECTRAL QUOTIENT = ", minimum(reva[csel]) / maximum(reva))
        end
    else
        println("Damping")
        display(-log.(abs.(eva[c_sel])) ./ abs.(angle.(eva[c_sel])))
        println("Frequency")
        display(angle.(eva[c_sel]) ./ Tstep)
        reva = log.(abs.(eva[c_sel]))
        println("R. SPECTRAL QUOTIENT = ", minimum(reva[SEL]) / maximum(reva))
        if !isempty(csel)
            println("S. SPECTRAL QUOTIENT = ", minimum(reva[csel]) / maximum(reva))
        end
    end
#     display([abs.(eva[sortperm(roughness)]);; roughness[sortperm(roughness)]])
    # right bundles
    XW = permutedims(eve_reshape[:,:,c_sel],(1,3,2))*sqrt(length(grid))
    XWre = zeros(eltype(B), size(XW)...)
    if ~isempty(i_sel)
        XWre[:,1:2:2*length(i_sel),:] .= real.(permutedims(eve_reshape[:,:,i_sel],(1,3,2))*sqrt(length(grid)*2))
        XWre[:,2:2:2*length(i_sel),:] .= imag.(permutedims(eve_reshape[:,:,i_sel],(1,3,2))*sqrt(length(grid)*2))
    end
    if ~isempty(r_sel)
        XWre[:,2*length(i_sel)+1:end,:] .= real.(permutedims(eve_reshape[:,:,r_sel],(1,3,2))*sqrt(length(grid)))
    end
#     MU = QPPolynomial(dim_out, dim_in, fourier_order, 1, 1, ℂ)
    XU = permutedims(left_eve_reshape[c_sel,:,:],(1,2,3))*sqrt(length(grid))

    XUre = zeros(eltype(B), size(XU)...)
    if ~isempty(i_sel)
        XUre[1:2:2*length(i_sel),:,:] .= real.(permutedims(left_eve_reshape[i_sel,:,:],(1,2,3))*sqrt(length(grid)*2))
        XUre[2:2:2*length(i_sel),:,:] .= -imag.(permutedims(left_eve_reshape[i_sel,:,:],(1,2,3))*sqrt(length(grid)*2))
    end
    if ~isempty(r_sel)
        XUre[2*length(i_sel)+1:end,:,:] .= real.(permutedims(left_eve_reshape[r_sel,:,:],(1,2,3))*sqrt(length(grid)))
    end
    # create an orthogonal version
    # orthogonality is separated to "SEL" and the complement "csel"
    XUreOrth = zeros(eltype(B), size(XU)...)
    XUreOrth[SEL,:,:] .= orthognaliseLeftBundle(XUre[SEL,:,:])
    if ~isempty(csel)
        XUreOrth[csel,:,:] .= orthognaliseLeftBundle(XUre[csel,:,:])
    end
    # normalise
    @tullio IV[i,j,k] := XU[i,p,k] * XW[p,j,k]
    @tullio IVre[i,j,k] := XUre[i,p,k] * XWre[p,j,k]
    @tullio IVreOrth[i,j,k] := XUreOrth[i,p,k] * XWre[p,j,k]
    XWreOrth = zeros(eltype(B), size(XW)...)
    for k=1:size(IV,3)
#         @show abs.(eigvals(IV[:,:,k]))
#         @show abs.(eigvals(IVre[:,:,k]))
        XW[:,:,k] .= XW[:,:,k]*inv(IV[:,:,k])
        XWre[:,:,k] .= XWre[:,:,k]*inv(IVre[:,:,k])
        XWreOrth[:,:,k] .= XWre[:,:,k]*inv(IVreOrth[:,:,k])
    end
    return XU, XW, XUre, XWre, XUreOrth, XWreOrth, eva # Lambda
end

# transforms the polynomial F(x,t), represented M, X into the coordinate system of XU0, XW0
# the transformation is
function bundleTransform(M::QPPolynomial{dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, ℝ}, X, 
                         XU0::AbstractArray{T,3}, XW0::AbstractArray{T,3}, omega, sel = 1:size(XU0,1)) where {dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, T}
    if T <: Complex
        field = ℂ
    else
        field = ℝ
    end
    MW = QPPolynomial(dim_in, length(sel), fourier_order, 1, 1, field)
    XW = zero(MW)
    SetLinearPart!(MW, XW, XW0[:,sel,:])
    # setting the U part
    MUsh = QPPolynomial(length(sel), dim_out, fourier_order, 1, 1, field)
    XU = zero(MUsh)
    SetLinearPart!(MUsh, XU, XU0[sel,:,:])
    # need to shift
    XUsh = zero(MUsh)
    SH = shiftOperator(getgrid(fourier_order), omega)
    @tullio XU0_SH[i,j,k] := XU0[i,j,l] * SH[l,k]
    SetLinearPart!(MUsh, XUsh, XU0_SH[sel,:,:])

    # make the transformation of the whole system into a diagonal form
    # Ush o F
    MUF = QPPolynomial(length(sel), dim_in, fourier_order, min_polyorder, max_polyorder, field)
    XUF = zero(MUF)
    if T <: Complex
        Mc = QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field)
        Mc.mexp .= M.mexp
#         QPPolynomialSubstitute!(MUF, XUF, MU, XU, Mc, Complex.(X), omega)
        QPPolynomialSubstitute!(MUF, XUF, MUsh, XUsh, Mc, Complex.(X))
    else
#         QPPolynomialSubstitute!(MUF, XUF, MU, XU, M, X, omega)
        QPPolynomialSubstitute!(MUF, XUF, MUsh, XUsh, M, X)
    end
    # Ush o F o W
    MUFW = QPPolynomial(length(sel), length(sel), fourier_order, min_polyorder, max_polyorder, field)
    XUFW = zero(MUFW)
    QPPolynomialSubstitute!(MUFW, XUFW, MUF, XUF, MW, XW)
    
    # checking   
#     println("bundle transform check")
#     A0 = GetLinearPart(M, X)
#     @tullio P0[i,j,k] := XU0[i,p,l] * SH[l,k] * A0[p,q,k] * XW0[q,j,k]
#     P1 = GetLinearPart(MUFW, XUFW)
#     @show P0
#     @show P1
#     @show norm(P0[sel,sel,:] - P1)
    
    return MW, XW, MUsh, XU, MUFW, XUFW
end

# transforms the polynomial F(x,t), represented M, X into the coordinate system of XU0, XW0
# the transformation is
function bundleTransform(M::QPPolynomial{dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, ℝ}, X, 
                         MU::QPPolynomial{dim_U, dim_out, fourier_order, Umin, Umax, fU}, XU::AbstractArray{T,3},
                         MW::QPPolynomial{dim_in, dim_W, fourier_order, Wmin, Wmax, fW}, XW::AbstractArray{T,3}, omega) where {dim_U, dim_out, dim_in, dim_W, fourier_order, min_polyorder, max_polyorder, Umin, Wmin, Umax, Wmax, fU, fW, T}
    if T <: Complex
        field = ℂ
    else
        field = ℝ
    end
    SH = shiftOperator(getgrid(fourier_order), omega)
    @tullio XUsh[i,j,k] := XU[i,j,l] * SH[l,k]

    # make the transformation of the whole system into a diagonal form
    # Ush o F
    MUF = QPPolynomial(dim_U, dim_in, fourier_order, min_polyorder, max_polyorder, field)
    XUF = zero(MUF)
    if T <: Complex
        Mc = QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field)
        Mc.mexp .= M.mexp
#         QPPolynomialSubstitute!(MUF, XUF, MU, XU, Mc, Complex.(X), omega)
        QPPolynomialSubstitute!(MUF, XUF, MU, XUsh, Mc, Complex.(X))
    else
#         QPPolynomialSubstitute!(MUF, XUF, MU, XU, M, X, omega)
        QPPolynomialSubstitute!(MUF, XUF, MU, XUsh, M, X)
    end
    # Ush o F o W
    MUFW = QPPolynomial(dim_U, dim_W, fourier_order, min_polyorder, max_polyorder, field)
    XUFW = zero(MUFW)
    QPPolynomialSubstitute!(MUFW, XUFW, MUF, XUF, MW, XW)
    
    # checking   
#     println("bundle transform check")
#     A0 = GetLinearPart(M, X)
#     @tullio P0[i,j,k] := XU0[i,p,l] * SH[l,k] * A0[p,q,k] * XW0[q,j,k]
#     P1 = GetLinearPart(MUFW, XUFW)
#     @show P0
#     @show P1
#     @show norm(P0[sel,sel,:] - P1)
    
    return MUFW, XUFW
end

function ODEbundleTransform(M::QPPolynomial{dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, ℝ}, X, 
                            XU0::AbstractArray{T,3}, XW0::AbstractArray{T,3}, omega, sel = 1:size(XU0,1)) where {dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, T}
    if T <: Complex
        field = ℂ
    else
        field = ℝ
    end
    MW = QPPolynomial(dim_in, length(sel), fourier_order, 1, 1, field)
    XW = zero(MW)
    SetLinearPart!(MW, XW, XW0[:,sel,:])
    MU = QPPolynomial(length(sel), dim_out, fourier_order, 1, 1, field)
    XU = zero(MU)
    SetLinearPart!(MU, XU, XU0[sel,:,:])

    # make the transformation of the whole system into a diagonal form
    # U o F
    MUF = QPPolynomial(length(sel), dim_in, fourier_order, min_polyorder, max_polyorder, field)
    XUF = zero(MUF)
    if T <: Complex
        Mc = QPPolynomial(dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, field)
        Mc.mexp .= M.mexp
        QPPolynomialSubstitute!(MUF, XUF, MU, XU, Mc, Complex.(X), 0.0)
    else
        QPPolynomialSubstitute!(MUF, XUF, MU, XU, M, X, 0.0)
    end
    # U o F o W
    MUFW = QPPolynomial(length(sel), length(sel), fourier_order, min_polyorder, max_polyorder, field)
    XUFW = zero(MUFW)
    QPPolynomialSubstitute!(MUFW, XUFW, MUF, XUF, MW, XW)
    
    XDW = thetaDerivative(MW, XW, omega)
    @tullio XUDW[i,j,k] := XU[i,p,k] * XDW[p,j,k]
    MUDW = QPPolynomial(length(sel), length(sel), fourier_order, min_polyorder, max_polyorder, field)
    XUDW = zero(MUDW)
    QPPolynomialSubstitute!(MUDW, XUDW, MU, XU, MW, XDW, 0.0)
    SetLinearPart!(MUFW, XUFW, GetLinearPart(MUFW, XUFW) - GetLinearPart(MUDW, XUDW))
    println("ODEbundleTransform")
    return MW, XW, MU, XU, MUFW, XUFW
end


# input: M, X a nonlinear QP map, 
#        omega, the phase shift in each iteration
#        epsilon, the separation between rings of eigenvalues that are considered a single cluster.
# output: Lambda, the eigenvalues that the linear QP system is reduced to
#         MW, XW, the right invariant vector bundles
#         MU, XU, the left invariant vector bundles
#         MUFW, XUFW the transformed system in the coordinate system of the invariant vector bundles
function smoothestDecomposition(M::QPPolynomial{dim_out, dim_in, fourier_order, min_polyorder, max_polyorder, ℝ}, X, omega, sel = 1:dim_in; ODE = false) where {dim_out, dim_in, fourier_order, min_polyorder, max_polyorder}
    @assert dim_in == dim_out "The polynomial should be X -> X and not X -> Y"
    B = GetLinearPart(M, X)
    # discard the real and orthogonal parts
    XUc0, XWc0, XUre, XWre, _, _, Lambda = vectorBundles(fourier_order, B, omega, ODE = ODE)
    if ODE
        MW, XW, MU, XU, MUFW, XUFW = ODEbundleTransform(M, X, XUc0, XWc0, omega, sel)
        return Lambda, MW, XW, MU, XU, MUFW, XUFW, XWre
    else
        MW, XW, MU, XU, MUFW, XUFW = bundleTransform(M, X, XUc0, XWc0, omega, sel)
        return Lambda, MW, XW, MU, XU, MUFW, XUFW, XWre
    end
end
