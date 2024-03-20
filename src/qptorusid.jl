
# finds a torus from a polynomial using Newton iteration
function torusId!(MK::QPConstant{dim_out, fourier_order, ℝ}, XK, MP::QPPolynomial{dim_out, dim_out, fourier_order, min_polyorder, max_polyorder, ℝ}, XP, omega) where {dim_out, fourier_order, min_polyorder, max_polyorder}
    grid = getgrid(fourier_order)
    BSH = bigShiftOperator(dim_out, grid, -omega)
    for k = 1:100
        MJ, XJ = Jacobian(MP, XP, MK, XK)
        TR = shiftMinusDiagonalOperator(grid, XJ, -omega)
        res = vec(Eval(MP, XP, collect(Diagonal(I,2*fourier_order+1)), XK)) .- BSH * vec(XK)
        @show norm(res)
        del = TR \ res
        XK .+= reshape(del, size(XK)...)
        if maximum(abs.(res)) <= 20*eps(eltype(res))
            return MJ, XJ
        end
    end
    return Jacobian(MP, XP, MK, XK)
end


function ODEtorusId!(MK::QPConstant{dim_out, fourier_order, ℝ}, XK, MP::QPPolynomial{dim_out, dim_out, fourier_order, min_polyorder, max_polyorder, ℝ}, XP, omega) where {dim_out, fourier_order, min_polyorder, max_polyorder}
    grid = getgrid(fourier_order)
    BSH = bigDifferentialOperator(dim_out, grid, omega)
    for k = 1:100
        MJ, XJ = Jacobian(MP, XP, MK, XK)
        TR = differentialMinusDiagonalOperator(grid, XJ, omega)
        res = BSH * vec(XK) .- vec(Eval(MP, XP, collect(Diagonal(I,2*fourier_order+1)), XK))
        @show norm(res)
        del = TR \ res
        XK .+= reshape(del, size(XK)...)
        if maximum(abs.(res)) <= 20*eps(eltype(res))
            return MJ, XJ
        end
    end
    return Jacobian(MP, XP, MK, XK)
end
