## SECTION Set-up
using InvariantModels
using CairoMakie
using GLMakie
using LaTeXStrings
using LinearAlgebra
GLMakie.activate!()

## Consider the Shaw-Pierre vector field
NDIM = 4

function onemass!(dz, z, p, t)
    kA = 1.0 # - p/2 * sin(t)^2 + p/4 * sin(t)
    kB = 2.5 # + p/2 * cos(t)^2 - p/4 * cos(t)
    cA = 0.06
    cB = 0.12
    dz .= [z[3],
           z[4],
           -(kA*z[1]) - 3*kB*z[1]^3 + 45*kB*z[1]^5 - 2*kB*z[1]*z[2] + 36*kB*z[1]^3*z[2] - kA*z[2]^2 + 6*kA*z[1]*z[2]^2 + 6*kB*z[1]*z[2]^2 - 36*kA*z[1]^2*z[2]^2 + 240*kA*z[1]^3*z[2]^2 - 360*kB*z[1]^3*z[2]^2 - 24*kB*z[1]*z[2]^3 + 9*kA*z[2]^4 - 180*kA*z[1]*z[2]^4 + 120*kB*z[1]*z[2]^4 - cA*z[3] - 6*cB*z[1]^2*z[3] + 120*cB*z[1]^4*z[3] + 48*cB*z[1]^2*z[2]*z[3] + 6*cA*z[2]^2*z[3] - 48*cA*z[1]*z[2]^2*z[3] + 360*cA*z[1]^2*z[2]^2*z[3] - 360*cB*z[1]^2*z[2]^2*z[3] - 120*cA*z[2]^4*z[3] - 2*cB*z[1]*z[4] + 24*cB*z[1]^3*z[4] - 2*cA*z[2]*z[4] + 6*cA*z[1]*z[2]*z[4] + 6*cB*z[1]*z[2]*z[4] - 24*cA*z[1]^2*z[2]*z[4] + 120*cA*z[1]^3*z[2]*z[4] - 360*cB*z[1]^3*z[2]*z[4] - 24*cB*z[1]*z[2]^2*z[4] + 24*cA*z[2]^3*z[4] - 360*cA*z[1]*z[2]^3*z[4] + 120*cB*z[1]*z[2]^3*z[4] + p*cos(t + pi/3),
           -(kB*z[1]^2) + 9*kB*z[1]^4 - kB*z[2] - 2*kA*z[1]*z[2] + 6*kA*z[1]^2*z[2] + 6*kB*z[1]^2*z[2] - 24*kA*z[1]^3*z[2] + 120*kA*z[1]^4*z[2] - 180*kB*z[1]^4*z[2] - 36*kB*z[1]^2*z[2]^2 - 3*kA*z[2]^3 + 36*kA*z[1]*z[2]^3 - 360*kA*z[1]^2*z[2]^3 + 240*kB*z[1]^2*z[2]^3 + 45*kA*z[2]^5 - 2*cB*z[1]*z[3] + 24*cB*z[1]^3*z[3] - 2*cA*z[2]*z[3] + 6*cA*z[1]*z[2]*z[3] + 6*cB*z[1]*z[2]*z[3] - 24*cA*z[1]^2*z[2]*z[3] + 120*cA*z[1]^3*z[2]*z[3] - 360*cB*z[1]^3*z[2]*z[3] - 24*cB*z[1]*z[2]^2*z[3] + 24*cA*z[2]^3*z[3] - 360*cA*z[1]*z[2]^3*z[3] + 120*cB*z[1]*z[2]^3*z[3] - cB*z[4] + 6*cB*z[1]^2*z[4] - 120*cB*z[1]^4*z[4] - 48*cB*z[1]^2*z[2]*z[4] - 6*cA*z[2]^2*z[4] + 48*cA*z[1]*z[2]^2*z[4] - 360*cA*z[1]^2*z[2]^2*z[4] + 360*cB*z[1]^2*z[2]^2*z[4] + 120*cA*z[2]^4*z[4] + p*cos(t)]
    return dz
end

function onemass(z, p, t)
    dz = zero(z)
    return onemass!(dz, z, p, t)
end

## Set up some system parameters
Amplitude_list = (0.0, 0.03) # forcing amplitude
omega_ode = 1.2 # Float64(pi) # forcing frequency

map_manifolds = true
create_data = false
train_model = false
load_pre_trained = false
data_driven = false
spectrum = false
# parameters of the numerical method
fourier_order_list = (0, 7) # fourier harmonics to be resolved
ODE_order = 7 # polynomial order of the calculations
SEL = [1 2] # which invariant vector bundle to use

# SECTION Set-up for ROM identification from data
# Importing a library to load/store data and specifying parameters
using BSON: @load, @save
# parameters for the data-driven part
maxSimAmp = 1.0     # maximum initial condition measured from the torus
maxAmp = 1.0        # filter out data with amplitude greater than maxAmp
dataRatio = 1.0     # the proportion of data to be retained when filtering (all of it)
R_order = 7         # polynomial order of R
U_order = 7         # polynomial order of U
V_order = 7         # polynomial order of V
S_order = 7         # polynomial order of S
STEPS = 800         # number of optimisation steps to use

# The results are then plotted
if data_driven
    fig = Figure(size=(1500, 375), fontsize=20)
    axDense = Axis(fig[1, 1], xscale=log10, xticks=LogTicks(WilkinsonTicks(4, k_min=3, k_max=5)))
    axErr = Axis(fig[1, 2], xscale=log10, xticks=LogTicks(WilkinsonTicks(3, k_min=4, k_max=4)))
    axFreq = Axis(fig[1, 3], xticks=WilkinsonTicks(3, k_min=3, k_max=4))
    axDamp = Axis(fig[1, 4], xticks=WilkinsonTicks(3, k_min=3, k_max=4))
    axDense.xlabel = "Data Density"
    axDense.ylabel = "Amplitude"
else
    fig = Figure(size=(1050, 375), fontsize=20)
    axErr = Axis(fig[1, 1], xscale=log10, xticks=LogTicks(WilkinsonTicks(4, k_min=4, k_max=5)))
    axFreq = Axis(fig[1, 2], xticks=WilkinsonTicks(3, k_min=3, k_max=4))
    axDamp = Axis(fig[1, 3], xticks=WilkinsonTicks(3, k_min=3, k_max=4))
end
axErr.xlabel = L"E_{\mathit{rel}}"
axErr.ylabel = "Amplitude"
axFreq.xlabel = "Frequency"
axFreq.ylabel = "Amplitude"
axDamp.xlabel = "Damping ratio"
axDamp.ylabel = "Amplitude"

# plotting
if 1 in SEL
    azimuth = 0.65
    dispMaxAmp = 0.15
    surf_lim_list = (0.1, 0.06)
    xlims!(axFreq, 0.9, 1.3)
    ylims!(axFreq, 0, dispMaxAmp)
    xlims!(axDamp, 0.0, 0.2)
    ylims!(axDamp, 0, dispMaxAmp)
else
    azimuth = 1.25
    dispMaxAmp = 0.1
    surf_lim_list = (0.1, 0.1)
    xlims!(axFreq, 1.54, 1.59)
    ylims!(axFreq, 0, dispMaxAmp)
    xlims!(axDamp, 0.036, 0.042)
    ylims!(axDamp, 0, dispMaxAmp)
end
xlims!(axErr, 1e-9, 1e-1)
ylims!(axErr, 0, dispMaxAmp)
if data_driven
    xlims!(axDense, 1e-2, 5)
    ylims!(axDense, 0, dispMaxAmp)
end

all_revision = "onemass-2p$(log2(scale_epsilon))-SIMAMP$(maxSimAmp)-F$(maximum(fourier_order_list))-R$(R_order)-U$(U_order)-V$(V_order)-S$(S_order)-MODE$(SEL[1])"

figEvArr = []
axTorArr = []
for (Amplitude, fourier_order, surf_lim) in zip(Amplitude_list, fourier_order_list, surf_lim_list)

# the names of data files
revision = "onemass-2p$(log2(scale_epsilon))-SIMAMP$(maxSimAmp)-AMP$(Amplitude)-F$(fourier_order)-R$(R_order)-U$(U_order)-V$(V_order)-S$(S_order)-MODE$(SEL[1])"
datarevision = "onemass-SIMAMP$(maxSimAmp)-AMP$(Amplitude)-F$(fourier_order)"

## SECTION Invariant Manifolds of Vector Fields
# Create a polynomial `MPode`, `XPode` out of our vector field
MPode = QPPolynomial(NDIM, NDIM, fourier_order, 0, ODE_order)
XPode = fromFunction(MPode, (x, t) -> onemass(x, Amplitude, t))

# The invariant manifold can now be calculated using [`QPODETorusManifold`](@ref):
MK, XK, MSn, XSn, MWn, XWn, MFdiag, XFdiag, MWd, XWd, XWre, oeva = QPODETorusManifold(MPode, XPode, omega_ode, SEL, threshold=0.1, resonance=false)

# The frequencies and damping ratios are extracted from the reduced order model using [`ODEFrequencyDamping`](@ref)
That, Rhat_r, rho, gamma = ODEFrequencyDamping(MWn, XWn, MSn, XSn, dispMaxAmp)
odeAmp = range(0, dispMaxAmp, length=1000)
odeFreq = abs.(That.(odeAmp))
odeDamp = -Rhat_r.(odeAmp) ./ odeFreq

if Amplitude > eps(1.0)
    pl_color = Makie.wong_colors()[4]
else
    pl_color = Makie.wong_colors()[3]
end

lines!(axFreq, odeFreq, odeAmp, label="ODE O($(ODE_order)) A=$(Amplitude)", color=pl_color, linestyle=:dash, linewidth=3)
lines!(axDamp, odeDamp, odeAmp, label="ODE O($(ODE_order)) A=$(Amplitude)", color=pl_color, linestyle=:dash, linewidth=3)
display(fig)

CairoMakie.activate!(type="svg")
save("fig-data-$(revision)-ODE.svg", fig)
GLMakie.activate!()

r_rg = range(0, dispMaxAmp, length=48)[2:end]
g_rg = getgrid(18)
t_rg = getgrid(18)
zz = reshape([r * exp(1im * g) for r in r_rg, g in g_rg, t in t_rg], 1, :)
dataZ = vcat(zz,conj.(zz))
thetaZ = vec([t for r in r_rg, g in g_rg, t in t_rg])
res, WoZ = InvariantModels.ODEManifoldAccuracy(MPode, XPode, MK, XK, MWn, XWn, MSn, XSn, omega_ode, thetaZ, dataZ)
amps = vec(sqrt.(sum(real.(WoZ .* conj.(WoZ)), dims=1)))
errs = vec(sqrt.(sum(real.(res .* conj.(res)), dims=1) ./ sum(real.(WoZ .* conj.(WoZ)), dims=1)))

hsU, errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX = ErrorStatistics(amps, errs, dispMaxAmp)

# lines!(axErr, errMaxX, errMaxY, color=pl_color, linestyle=:solid, linewidth=3)
# lines!(axErr, errMinX, errMinY, color=pl_color, linestyle=:solid, linewidth=3)
lines!(axErr, errMeanX, errMeanY, color=pl_color, linestyle=:dash, linewidth=3)
# lines!(axErr, errMeanX .+ errStdX, errMeanY, color=pl_color, linestyle=:dot, linewidth=3)

# continue
if spectrum
    sel_eva = diag(InvariantModels.GetLinearPart(MFdiag, XFdiag)[:,:,fourier_order+1])
    figEv = Figure(size=(1000,300))
    push!(figEvArr, figEv)
    axTor = Axis3(figEv[1:3,1], protrusions = 40, title = "a)", titlesize = 20, yticks=WilkinsonTicks(3, k_min=2, k_max=3), azimuth=azimuth)
    push!(axTorArr, axTor)
    axEv = Axis(figEv[1:3,2], aspect = 1.5, title = "b)", titlesize = 20, yticks=WilkinsonTicks(3, k_min=2, k_max=3))
    axV1 = Axis(figEv[1,3], aspect = 5, title = "c)", titlesize = 20, yticks=WilkinsonTicks(3, k_min=2, k_max=3))
    axV2 = Axis(figEv[2,3], aspect = 5, yticks=WilkinsonTicks(3, k_min=2, k_max=3))
    axV3 = Axis(figEv[3,3], aspect = 5, yticks=WilkinsonTicks(3, k_min=2, k_max=3))

    tm = collect(range(0,2*pi,length=100))
    V = [I[1,k]*1.0 for k in 1:NDIM, j in 1:100]
    torus = Eval(MK, XK, tm, V)
    pm = sortperm(vec(maximum(abs.(torus), dims=2)), rev=true)
    lines!(axTor, torus[pm[1],:], torus[pm[2],:], torus[pm[3],:])
    axTor.xlabel = latexstring("\$x_$(pm[1])\$")
    axTor.ylabel = latexstring("\$x_$(pm[2])\$")
    axTor.zlabel = latexstring("\$x_$(pm[3])\$")
    axTor.xlabelsize = 20
    axTor.ylabelsize = 20
    axTor.zlabelsize = 20

    lines!(axEv, [0, 0], [-maximum(imag.(oeva)), maximum(imag.(oeva))] , linestyle=:dash, linewidth=1)
    scatter!(axEv, real.(oeva), imag.(oeva))
    scatter!(axEv, real.(sel_eva[setdiff(1:length(sel_eva),SEL[1])]), imag.(sel_eva[setdiff(1:length(sel_eva),SEL[1])]), marker = :star5, markersize = 20)
    scatter!(axEv, real.(sel_eva[SEL[1]]), imag.(sel_eva[SEL[1]]), marker = :utriangle, markersize = 20)
    axEv.xlabel = L"\Re{\lambda}"
    axEv.ylabel = L"\Im{\lambda}"
    axEv.xlabelsize = 20
    axEv.ylabelsize = 20

    XWnlin = zero(MWn)
    InvariantModels.SetLinearPart!(MWn, XWnlin, InvariantModels.GetLinearPart(MWn, XWn))
    display([I[k,1] for k=1:length(SEL), l=1:length(tm)] .+ 0.0im)
    vecs = real.(Eval(MWn, XWnlin, tm .+ 0.0im, [I[k,1] for k=1:length(SEL), l=1:length(tm)] .+ 0.0im))

    # vecs = real.(Eval(MWd, XWd, tm, V))
    lines!(axV1, tm, vecs[pm[1],:])
    axV1.ylabel = latexstring("\$x_$(pm[1])\$")
    axV1.ylabelsize = 20
    lines!(axV2, tm, vecs[pm[2],:])
    axV2.ylabel = latexstring("\$x_$(pm[2])\$")
    axV2.ylabelsize = 20
    lines!(axV3, tm, vecs[pm[3],:])
    axV3.ylabel = latexstring("\$x_$(pm[3])\$")
    axV3.xlabel = L"\theta"
    axV3.ylabelsize = 20
    axV3.xlabelsize = 20
    resize_to_layout!(figEv)

    # manifold
    tsel = 52
    # ZZ = [r*exp(1im*p) for r in range(0,0.2,length=10), p in range(0,2*pi,length=12)]
    ZZ = [zr + 1im*zi for zr in range(-surf_lim,surf_lim,length=12), zi in range(-surf_lim,surf_lim,length=12)]
    Zg = collect(transpose(hcat(vec(ZZ), conj.(vec(ZZ)))))
    Sf = Eval(MWn, XWn, fill(tm[tsel]+0im,prod(size(ZZ))), Zg) .+ reshape(torus[:,tsel],:,1)
    x1 = real.(reshape(Sf[pm[1],:], size(ZZ)...))
    x2 = real.(reshape(Sf[pm[2],:], size(ZZ)...))
    x3 = real.(reshape(Sf[pm[3],:], size(ZZ)...))
    surface!(axTor, x1, x2, x3, colormap = [Makie.wong_colors()[7]], alpha = 0.4)
    scatter!(axTor, torus[pm[1],tsel], torus[pm[2],tsel], torus[pm[3],tsel], markersize = 20, color = Makie.wong_colors()[6])

    CairoMakie.activate!(type="svg")
    save("fig-spectrum-$(revision).svg", figEv)
    GLMakie.activate!()
end # spectrum
    
# continue
# SECTION Invariant Manifolds of Maps
# First we set up some additional parameters, such as the sampling period ``\Delta t`` = `Tstep`
Tstep = 0.8
omega = omega_ode * Tstep

if map_manifolds
    # We now create a discrete-time map from the vector field by Taylor expanding an ODE solver about the origin using [`mapFromODE`](@ref). We take 500 time steps on the interval ``[\theta,\theta + \Delta t]``. The resulting map `MP`, `XPmap` is dependent on the phase variable ``\theta \in [0,2\pi)``
    MP = QPPolynomial(NDIM, NDIM, fourier_order, 0, ODE_order)
    XPmap = zero(MP)
    mapFromODE(MP, XPmap, onemass!, Amplitude, omega_ode, Tstep / 500, Tstep)

    # The invariant manifold of the map is calculated using [`QPMAPTorusManifold`](@ref)
    MK, XK, MSn, XSn, MWn, XWn, MSd, XSd = QPMAPTorusManifold(MP, XPmap, omega, SEL, threshold = 0.1, resonance = false)

    # The frequencies and damping ratios are extracted from the reduced order model using [`MAPFrequencyDamping`](@ref)
    That, Rhat_r, rho, gamma = MAPFrequencyDamping(MWn, XWn, MSn, XSn, dispMaxAmp)
    mapAmp = range(0, dispMaxAmp, length=1000)
    mapFreq = abs.(That.(mapAmp)) ./ Tstep
    mapDamp = -log.(abs.(Rhat_r.(mapAmp))) ./ abs.(That.(mapAmp))

    # The results are plotted
    lines!(axFreq, mapFreq, mapAmp, label="MAP O($(ODE_order)) A=$(Amplitude)", color=pl_color, linestyle=:dashdot, linewidth=3)
    lines!(axDamp, mapDamp, mapAmp, label="MAP O($(ODE_order)) A=$(Amplitude)", color=pl_color, linestyle=:dashdot, linewidth=3)
    display(fig)

    res, WoZ = InvariantModels.MAPManifoldAccuracy(MP, XPmap, MK, XK, MWn, XWn, MSn, XSn, omega, thetaZ, dataZ)
    amps = vec(sqrt.(sum(real.(WoZ .* conj.(WoZ)), dims=1)))
    errs = vec(sqrt.(sum(real.(res .* conj.(res)), dims=1) ./ sum(real.(WoZ .* conj.(WoZ)), dims=1)))

    hsU, errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX = ErrorStatistics(amps, errs, dispMaxAmp)

#     lines!(axErr, errMaxX, errMaxY, color=pl_color, linestyle=:solid, linewidth=3)
#     lines!(axErr, errMinX, errMinY, color=pl_color, linestyle=:solid, linewidth=3)
    lines!(axErr, errMeanX, errMeanY, color=pl_color, linestyle=:dashdot, linewidth=3)
#     lines!(axErr, errMeanX .+ errStdX, errMeanY, color=pl_color, linestyle=:dot, linewidth=3)
    display(fig)
    
    CairoMakie.activate!(type="svg")
    save("fig-data-$(revision)-MAP.svg", fig)
    GLMakie.activate!()
    
    # SECTION Invariant Foliations of Maps
    # Using [`QPGraphStyleFoliations`](@ref), two invariant foliations are calculated and the invariant manifold defined by the zero level-set of the second foliation is extracted.
    MRf, XRf, MWf, XWf, MRt, XRt, MUt, XUt, MSt, XSt, MVt, XVt = QPGraphStyleFoliations(MP, XPmap, omega, SEL; dataScale=1, resonance=false, threshold=0.1)
    # The results are plotted
#     MWf, XWf = toFourier(MW, XW)
    That, Rhat_r, rho, gamma = MAPFrequencyDamping(MWf, XWf, MRf, XRf, dispMaxAmp)
    foilAmp = range(0, dispMaxAmp, length=1000)
    foilFreq = abs.(That.(foilAmp)) ./ Tstep
    foilDamp = -log.(abs.(Rhat_r.(foilAmp))) ./ abs.(That.(foilAmp))
    if data_driven
        lines!(axFreq, foilFreq, foilAmp, label="FOIL O($(ODE_order)) A=$(Amplitude)", color=pl_color, linestyle=Linestyle([1.0, 3.0, 4.0, 6.0]), linewidth=3)
        lines!(axDamp, foilDamp, foilAmp, label="FOIL O($(ODE_order)) A=$(Amplitude)", color=pl_color, linestyle=Linestyle([1.0, 3.0, 4.0, 6.0]), linewidth=3)
    else
        lines!(axFreq, foilFreq, foilAmp, label="FOIL O($(ODE_order)) A=$(Amplitude)", color=pl_color, linestyle=:solid, linewidth=3)
        lines!(axDamp, foilDamp, foilAmp, label="FOIL O($(ODE_order)) A=$(Amplitude)", color=pl_color, linestyle=:solid, linewidth=3)
    end
    display(fig)

    res, WoZ = InvariantModels.MAPManifoldAccuracy(MP, XPmap, MK, XK, MWf, XWf, MRf, XRf, omega, thetaZ, dataZ)
    amps = vec(sqrt.(sum(real.(WoZ .* conj.(WoZ)), dims=1)))
    errs = vec(sqrt.(sum(real.(res .* conj.(res)), dims=1) ./ sum(real.(WoZ .* conj.(WoZ)), dims=1)))

    hsU, errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX = ErrorStatistics(amps, errs, dispMaxAmp)

#     println("--!!!!!!!!!!!!!!!!!!! FOIL ERR PLOT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     @show errMaxX, errMaxY
#     lines!(axErr, errMaxX, errMaxY, color=:red, linestyle=:solid, linewidth=4)
#     lines!(axErr, errMinX, errMinY, color=:red, linestyle=:solid, linewidth=4)
    lines!(axErr, errMeanX, errMeanY, color=pl_color, linestyle=:solid, linewidth=3)
#     lines!(axErr, errMeanX .+ errStdX, errMeanY, color=:red, linestyle=:dot, linewidth=4)
    display(fig)
#     println("++!!!!!!!!!!!!!!!!!!! FOIL ERR PLOT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#     @show XRf
    
    CairoMakie.activate!(type="svg")
    save("fig-data-$(revision)-FOIL.svg", fig)
    GLMakie.activate!()
end

continue

# creating data
# XWre defines the ellipsiodal distribution of initial conditions. Here we make it spherical by setting XWre the identity
XWre = zeros(NDIM, NDIM, 2*fourier_order+1)
for k in axes(XWre,3)
    XWre[:,:,k] .= Diagonal(I,size(XWre,1))
end

if create_data
    # the actual data generation
    dataX, dataY, thetaX, thetaY, thetaNIX, thetaNIY = generate(NDIM, onemass!, ones(NDIM) * maxSimAmp, 600, 50, fourier_order, omega_ode, Tstep, Amplitude, XWre)
    @save "data-$(datarevision).bson" dataX dataY thetaX thetaY Tstep
else
    @load "data-$(datarevision).bson" dataX dataY thetaX thetaY Tstep
end

# finding the linear model
A, b, MK1, XK1 = findLinearModel(dataX, thetaX, dataY, thetaY, omega)

dataIdV = Array{Any,1}(undef, 1)
# if the model is pre-trained or training is not needed
if load_pre_trained
    println("LOADING PRE-TRAINED MODEL")
    @load "CF-$(revision)-pre.bson" MCF XCF Tstep dataId dataScale
    dataIdV[1] = dataId
    thetaTX, dataTX, thetaTY, dataTY, dataScale__, preId, R1, S1, W1 = QPPreProcess(XK1, A, omega, thetaX, dataX, thetaY, dataY, SEL; Tstep=Tstep, maxAmp=maxAmp, data_ratio=dataRatio, preId=dataIdV[1], dataScale=dataScale)
elseif !train_model
    println("LOADING MODEL")
    @load "CF-$(revision).bson" MCF XCF Tstep dataId dataScale
    dataIdV[1] = dataId
    thetaTX, dataTX, thetaTY, dataTY, dataScale__, preId, R1, S1, W1 = QPPreProcess(XK1, A, omega, thetaX, dataX, thetaY, dataY, SEL; Tstep=Tstep, maxAmp=maxAmp, data_ratio=dataRatio, preId=dataIdV[1], dataScale=dataScale)
else
    println("MODEL FROM SCRATCH")
    # transform the data into the new coordinate system
    thetaTX, dataTX, thetaTY, dataTY, dataScale, preId, R1, S1, W1 = QPPreProcess(XK1, A, omega, thetaX, dataX, thetaY, dataY, SEL; Tstep=Tstep, maxAmp=maxAmp)
    # setting up the data structures of the foliation
    MCF, XCF = QPCombinedFoliation(NDIM, 2, fourier_order, R_order, U_order, S_order, V_order, R1, S1, MK1, zero(XK1), sparse=false)
    dataIdV[1] = preId
end

# creating a cache
XCFcache = makeCache(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY)

if train_model
    radii=to_zero(XCF)
    QPOptimise(MCF, XCF, thetaTX, dataTX, thetaTY, dataTY;
            maxit=8, gradient_ratio=2^(-7), gradient_stop=2^(-29), steps=STEPS, name=revision,
            cache=XCFcache, dataScale=dataScale,
            omega=omega, Tstep=Tstep, dataId=dataIdV[1], radii=radii)
end

MSn, XSn, MFW, XFWoWdoWn, MW, XW = QPPostProcess(MCF, XCF, W1, omega)

That, Rhat_r, rho, gamma = MAPFrequencyDamping(MFW, XFWoWdoWn, MSn, XSn, dispMaxAmp / dataScale)
r_data = range(0, dispMaxAmp / dataScale, length=1000)
dataFreq = abs.(That.(r_data)) / Tstep
dataDamp = -log.(abs.(Rhat_r.(r_data))) ./ abs.(That.(r_data))
dataAmp = r_data .* dataScale
# plotting
lines!(axFreq, dataFreq, dataAmp, label="DATA O($(R_order), $(S_order)) A=$(Amplitude)", color=pl_color, linestyle=:solid, linewidth=3)
lines!(axDamp, dataDamp, dataAmp, label="DATA O($(R_order), $(S_order)) A=$(Amplitude)", color=pl_color, linestyle=:solid, linewidth=3)

CairoMakie.activate!(type="svg")
save("fig-data-$(revision)-DATA.svg", fig)
GLMakie.activate!()

OnManifoldAmplitude, hsU, errMaxX, errMaxY, errMinX, errMinY, errMeanX, errMeanY, errStdX = ErrorStatistics(MCF, XCF, MW, XW, thetaTX, dataTX, thetaTY, dataTY; dataScale=dataScale, maxAmp=dispMaxAmp, cache=XCFcache)
den = Makie.KernelDensity.kde(sort(vcat(OnManifoldAmplitude, -OnManifoldAmplitude)))
atol = eps(maximum(den.density))
den.density[findall(isapprox.(den.density, 0, atol=atol))] .= atol
dataDensityX = den.density
dataDensityY = den.x

lines!(axDense, dataDensityX, dataDensityY, color=pl_color)
# heatmap!(axErr, hsU, colormap=GLMakie.Reverse(:greys))
lines!(axErr, errMaxX, errMaxY, color=pl_color, linestyle=:solid, linewidth=3)
lines!(axErr, errMinX, errMinY, color=pl_color, linestyle=:solid, linewidth=3)
lines!(axErr, errMeanX, errMeanY, color=pl_color, linestyle=:dash, linewidth=3)
lines!(axErr, errMeanX .+ errStdX, errMeanY, color=pl_color, linestyle=:dot, linewidth=3)

@save "result-$(revision).bson" odeFreq odeDamp odeAmp mapFreq mapDamp mapAmp foilFreq foilDamp foilAmp dataFreq dataDamp dataAmp OnManifoldAmplitude hsU errMaxX errMaxY errMinX errMinY errMeanX errMeanY errStdX

end # for (Amplitude, fourier_order) in ...

# creating the legend
if data_driven
    fig[1, 5] = Legend(fig, axFreq, merge=true, unique=true, labelsize=16, backgroundcolor=(:white, 0), framevisible=false, rowgap=1)
    if SEL[1] in [1 2]
        text!(axDense, "a)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axErr, "b)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axFreq, "c)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axDamp, "d)", space=:relative, position=Point2f(0.1, 0.9))
    else
        text!(axDense, "e)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axErr, "f)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axFreq, "g)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axDamp, "h)", space=:relative, position=Point2f(0.1, 0.9))
    end
else
    fig[1, 4] = Legend(fig, axFreq, merge=true, unique=true, labelsize=16, backgroundcolor=(:white, 0), framevisible=false, rowgap=1)
    if SEL[1] in [1 2]
        text!(axErr, "a)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axFreq, "b)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axDamp, "c)", space=:relative, position=Point2f(0.1, 0.9))
    else
        text!(axErr, "d)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axFreq, "e)", space=:relative, position=Point2f(0.1, 0.9))
        text!(axDamp, "f)", space=:relative, position=Point2f(0.1, 0.9))
    end
end

resize_to_layout!(fig)
display(fig)

CairoMakie.activate!(type="svg")
save("fig-data-$(all_revision).svg", fig)
GLMakie.activate!()
