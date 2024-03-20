function exposcale(p, delta, len)
    alpha = delta^(p - 1)
    pt = range(0, 1, length=len + 1)
    thr = floor(delta * len)
    sc1 = alpha * (1:thr) / len
    sc2 = ((thr+1:len) / len) .^ p
    return vcat(sc1, sc2)
end

@doc raw"""
    generate(NDIM, rhs!, maxIC, nruns, npoints, fourier_order, omega, Tstep, A, XWre)
    
Creates a data set from a differential equation, represented by `rhs!`. An invariant torus is found by observing a convergent simulation to the torus. The initial conditions are sampled from an ellipsoid about the torus. The directions of the ellipsoid are given by the linear transformation `XWre`. For each column of `XWre` the maximum amplitude is specified by the corresponding element of the vector `maxIC`.

The input parameters are
* `NDIM` the dimensionality of the system
* `rhs!` the vector field in the form of `fun(dx, x, p, theta)`, where `x` is the input state var, `dx` is the time-derivative ``\frac{\mathrm{d}\boldsymbol{x}}{\mathrm{d}t}``, `p` is the parameter, which equals to `par` and `theta` is the phase of forcing on the interval ``[0, 2\pi)``.
* `maxIC` is a vector of size `NDIM`, containing the maximum simulation amplitudes in each direction of `XWre`
* `nruns` the number of simulation runs
* `npoints` the length of the trajectory for each simulation run
* `fourier_order` the number of Fourier harmonics to consider when calculating ``\boldsymbol{\Theta}_x``, ``\boldsymbol{\Theta}_y``
* `omega` the forcing frequency, as in ``\dot{\theta} = \omega``
* `Tstep` the time step between two samples of the trajectory, denoted by ``\Delta t``
* `A` the parameter `p` of the vector field
* `XWre` is the transformation that defines the axes of the ellipsoid about the invariant torus
    
The function returns
```
dataX, dataY, thetaX, thetaY, thetaNIX, thetaNIY
```
* `dataX` is matrix ``\boldsymbol{X}``. Each column corresponts to a data point.
* `dataY` is matrix ``\boldsymbol{Y}``.
* `thetaX` is matrix ``\boldsymbol{\Theta}_x``
* `thetaY` is matrix ``\boldsymbol{\Theta}_y``
* `thetaNIX` is the vector ``(\theta_1, \ldots, \theta_N)``
* `thetaNIY` is the vector ``(\theta_1 + \omega \Delta t, \ldots, \theta_N + \omega \Delta t)``
space
"""
function generate(NDIM, rhs!, maxIC, nruns, npoints, fourier_order, omega, Tstep, A, XWre)
    u0 = zeros(NDIM)
    grid = getgrid(fourier_order)
    # hope it converges...
    period = 2 * pi / omega
    tspan = (0, 4000 * period)
    prob = ODEProblem((x, y, p, t) -> rhs!(x, y, p, omega * t), u0, tspan, A)
    sol = solve(prob, Vern7(), abstol=1e-8, reltol=1e-8)
    tend = (floor(tspan[end] / period) - 2) * period
    persol = Fun(t -> sol(tend + t), 0 .. period)
    # testing periodicity
    ts = range(0.0, period, length=100)
    a = sol(tend .+ ts)
    b = sol((tend - period) .+ ts)
    @show size(b)
    d = sqrt.(sum(a .^ 2, dims=1))
    println("Periodicity of solution = ", norm(a - b), " norm of solution = ", maximum(d))

    function condition(u, t, integrator)
        norm(u - persol(mod(t, period))) > 200 * maximum(maxIC)
    end
    function affect!(integrator)
        println("TERMINATED")
        terminate!(integrator)
    end
    # END PARAMETERS
    zs = Array{Array{Float64},1}(undef, nruns)
    ts = Array{Array{Float64},1}(undef, nruns)
    # setting initial conditions
    aa = randn(NDIM, nruns)
    ics = aa ./ sqrt.(sum(aa .^ 2, dims=1)) .* reshape(exposcale(2 / NDIM, 0.1, nruns), 1, nruns)
    t0s = rand(nruns)
    for j = 1:nruns
        shift = period * t0s[j]
        theta0 = fourierInterplate(grid, mod(shift * omega, 2 * pi)) # same as t0s[j] * 2 * Pi
        ic0 = ics[:, j]
        @tullio u0disp[i] := XWre[i, j, k] * maxIC[j] * ic0[j] * theta0[k]
        u0 = persol(shift) + u0disp
        #         @show persol(shift), u0disp, u0
        tspan = (shift, Tstep * (npoints + 1) + shift) # 51 intervals with T=0.8 as in Proc Roy Soc Paper
        prob = ODEProblem((x, y, p, t) -> rhs!(x, y, p, omega * t), u0, tspan, A, callback=DiscreteCallback(condition, affect!))

        sol = solve(prob, Vern7(), abstol=1e-8, reltol=1e-8)
        trange = range(start=sol.t[1], stop=sol.t[end], step=Tstep)
        #             @show length(1+(j-1)*npoints:j*npoints), size([sum(sol(t))/length(sol(t)) for t in trange])
        zs[j] = reduce(hcat, sol(trange).u)
        ts[j] = collect(trange) .* omega
        #         @show j, nruns
    end

    npt = sum([size(a, 2) - 1 for a in zs])
    dataX = zeros(NDIM, npt)
    dataY = zeros(NDIM, npt)
    thetaX = zeros(length(grid), npt)
    thetaY = zeros(length(grid), npt)
    thetaNIX = zeros(npt)
    thetaNIY = zeros(npt)

    let it = 1
        for k in eachindex(zs)
            #         @show size(dataX[:,it:it+size(zs[k],2)-2]), size(zs[k][:,1:end-1])
            dataX[:, it:it+size(zs[k], 2)-2] .= zs[k][:, 1:end-1]
            dataY[:, it:it+size(zs[k], 2)-2] .= zs[k][:, 2:end]
            thetaX[:, it:it+size(zs[k], 2)-2] .= fourierInterplate(grid, mod.(ts[k][1:end-1], 2 * pi))
            thetaY[:, it:it+size(zs[k], 2)-2] .= fourierInterplate(grid, mod.(ts[k][2:end], 2 * pi))
            thetaNIX[it:it+size(zs[k], 2)-2] .= mod.(ts[k][1:end-1], 2 * pi)
            thetaNIY[it:it+size(zs[k], 2)-2] .= mod.(ts[k][2:end], 2 * pi)
            it = it + size(zs[k], 2) - 1
        end
    end
    return dataX, dataY, thetaX, thetaY, thetaNIX, thetaNIY
end

# MF, XF is the map in polynomial form. Produced using integrating Taylor seies
# MK, XK is the steady state torus that was solved for before
@doc raw"""
    generateMap(MF, XF, MK, XK, maxIC, nruns, npoints, fourier_order, omega, Tstep)

Similar to [`generate`](@ref), except that it simulates a discrete-time system.

The inputs differ from [`generate`](@ref). Instead of the vector field, the following must be specified
* `MF`, `XF` the discrete-time map of the system
* `MK`, `XK` the invariant torus arbout which the system is simulated.
There is no way the specify an ellipsoid for initial conditions, they are taken from a sphere of radius `maxIC`.
"""
function generateMap(MF, XF, MK, XK, maxIC, nruns, npoints, fourier_order, omega, Tstep)
    # END PARAMETERS
    zs = Array{Array{Float64},1}(undef, nruns)
    ts = Array{Array{Float64},1}(undef, nruns)
    # setting initial conditions
    NDIM = size(XK, 1)
    aa = randn(NDIM, nruns)
    ics = aa ./ sqrt.(sum(aa .^ 2, dims=1)) .* reshape(exposcale(2 / NDIM, 0.1, nruns), 1, nruns)
    t0s = rand(nruns)
    for j = 1:nruns
        t0 = 2 * pi * t0s[j]
        u0 = Eval(MK, XK, t0) + ics[:, j] * maxIC
        trange = zeros(npoints + 1)
        sol = zeros(length(u0), npoints + 1)
        trange[1] = t0
        sol[:, 1] = u0
        k = 1
        while k <= npoints
            trange[k+1] = mod(trange[k] + Tstep * omega, 2 * pi)
            sol[:, k+1] .= Eval(MF, XF, trange[k], sol[:, k])
            if norm(sol[:, k+1] - Eval(MK, XK, trange[k+1])) > 20 * maxIC
                println("TERMINATED")
                break
            else
                k = k + 1
            end
        end
        zs[j] = sol[:, 1:k]
        ts[j] = trange[1:k]
        #         @show j, nruns
    end

    grid = getgrid(fourier_order)
    npt = sum([size(a, 2) - 1 for a in zs])
    dataX = zeros(NDIM, npt)
    dataY = zeros(NDIM, npt)
    thetaX = zeros(length(grid), npt)
    thetaY = zeros(length(grid), npt)
    thetaNIX = zeros(npt)
    thetaNIY = zeros(npt)

    let it = 1
        for k in eachindex(zs)
            #         @show size(dataX[:,it:it+size(zs[k],2)-2]), size(zs[k][:,1:end-1])
            dataX[:, it:it+size(zs[k], 2)-2] .= zs[k][:, 1:end-1]
            dataY[:, it:it+size(zs[k], 2)-2] .= zs[k][:, 2:end]
            thetaX[:, it:it+size(zs[k], 2)-2] .= fourierInterplate(grid, ts[k][1:end-1])
            thetaY[:, it:it+size(zs[k], 2)-2] .= fourierInterplate(grid, ts[k][2:end])
            thetaNIX[it:it+size(zs[k], 2)-2] .= mod.(ts[k][1:end-1], 2 * pi)
            thetaNIY[it:it+size(zs[k], 2)-2] .= mod.(ts[k][2:end], 2 * pi)
            it = it + size(zs[k], 2) - 1
        end
    end
    return dataX, dataY, thetaX, thetaY, thetaNIX, thetaNIY
end
