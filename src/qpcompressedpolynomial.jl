struct QPCompressedPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,𝔽} <: AbstractManifold{𝔽}
    perp_idx
    M::AbstractManifold{𝔽}
    R::AbstractRetractionMethod
    VT::AbstractVectorTransportMethod
end

# the constant and linear terms need to be represented separately as is done in qpfoliation.jl
# the projection is handled in
function QPCompressedPolynomial(dim_out, dim_in, perp_idx, fourier_order, min_polyorder, max_polyorder, field::AbstractNumbers=ℝ; node_rank=max(6, 2*fourier_order+1))
    @assert min_polyorder >= 2 "A compressed polynomial cannot have constant or linear terms, the lowest order is quadratic."
    indices = unique(sort(perp_idx))
    @assert issubset(indices, 1:dim_in) "perp_idx does not belong to 1:dim_in ."
    mlist = tuple([HTTensor(vcat(2 * fourier_order + 1, length(perp_idx), repeat([dim_in], k - 1)), dim_out, node_rank=node_rank, max_rank = 24) for k = min_polyorder:max_polyorder]...)
    M = ProductManifold(mlist...)
    R = ProductRetraction(map(x -> getfield(x, :R), mlist)...)
    VT = ProductVectorTransport(map(x -> getfield(x, :VT), mlist)...)
    return QPCompressedPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}(indices, M, R, VT)
end

function copyMost!(MD::QPCompressedPolynomial, XD, MS::QPCompressedPolynomial, XS)
    XD .= XS
    nothing
end

# BOILERPLATE
function zero(M::QPCompressedPolynomial)
    return ArrayPartition(map(zero, M.M.manifolds))
end

function zero(M::ProductManifold)
    return ArrayPartition(map(zero, M.manifolds))
end

function zero(M::Euclidean{N,ℝ}) where {N}
    return zeros(Manifolds.get_parameter(M.size)...)
    #     return zeros(N.parameters...)
end

function zero(M::Stiefel{N,ℝ}) where {N}
    A = zeros(Manifolds.get_parameter(M.size)...)
    k = Manifolds.get_parameter(M.size)[2]
    A[1:k, :] .= Diagonal(I, k)
    return A
end

function randn(M::QPCompressedPolynomial)
    return ArrayPartition(map(randn, M.M.manifolds))
end

function randn(M::ProductManifold)
    return ArrayPartition(map(randn, M.manifolds))
end

function randn(M::Euclidean{N,𝔽}, p=nothing) where {N,𝔽}
    return randn(Manifolds.get_parameter(M.size)...)
end

function randn(M::Stiefel{N,ℝ}) where {N}
    n, k = Manifolds.get_parameter(M.size)
    A = zeros(n, k)
    A[randperm(n)[1:k], :] .= Diagonal(I, k)
    return A
end

function randn(M::Stiefel{N,ℝ}, p) where {N}
    return rand(M; vector_at=p)
end

function zero_vector!(M::QPCompressedPolynomial, X, p)
    return zero_vector!(M.M, X, p)
end

function manifold_dimension(M::QPCompressedPolynomial)
    return manifold_dimension(M.M)
end

function inner(M::QPCompressedPolynomial, p, X, Y)
    return inner(M.M, p, X, Y)
end

function project!(M::QPCompressedPolynomial, Y, q, X)
    return project!(M.M, Y, q, X)
end

function retract!(M::QPCompressedPolynomial, q, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    return retract!(M.M, q, p, X, t, method)
end

function retract(M::QPCompressedPolynomial, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    return retract(M.M, p, X, t, method)
end

function vector_transport_to!(M::QPCompressedPolynomial, Y, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::QPCompressedPolynomial, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to(M.M, p, X, q, method)
end

function get_coordinates(M::QPCompressedPolynomial, p, X, B::AbstractBasis)
    return get_coordinates(M.M, p, X, B)
end

function get_vector(M::QPCompressedPolynomial, p, c, B::AbstractBasis)
    return get_vector(M.M, p, c, B)
end

function InterpolationWeights(M::QPCompressedPolynomial{dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}, theta) where {dim_out,dim_in,fourier_order,min_polyorder,max_polyorder,field}
    grid = getgrid(fourier_order)
    return fourierInterplate(grid, theta)
end

# END BOILERPLATE

function makeCache(M::QPCompressedPolynomial, X, theta::Matrix, data::Matrix)
    # Eval for a HTtensor takes VarArg arguments, so we can write thata, data
    return ArrayPartition(map((m, x) -> makeCache(m, x, theta, view(data, M.perp_idx, :), data), M.M.manifolds, X.x))
end

function updateCache!(cache, M::QPCompressedPolynomial, X, theta::Matrix, data::Matrix)
    for k = 1:length(cache.x)
        updateCache!(cache.x[k], M.M.manifolds[k], X.x[k], theta, view(data, M.perp_idx, :), data)
    end
    return nothing
end

function updateCache!(cache, M::QPCompressedPolynomial, X, theta::Matrix, data::Matrix, sel)
    updateCachePartial!(cache.x[sel[1]], M.M.manifolds[sel[1]], X.x[sel[1]], theta, view(data, M.perp_idx, :), data, ii=sel[2])
    return nothing
end

function Eval(M::QPCompressedPolynomial, X, theta::Matrix, data::Matrix; cache=makeCache(M, X, theta, data))
    # Eval for a HTtensor takes VarArg arguments, so we can write thata, data
    return mapreduce((m, x, c) -> Eval(m, x, theta, view(data, M.perp_idx, :), data, DV=c), .+, M.M.manifolds, X.x, cache.x)
end

# pointless to cache, the data always changes
function Eval(M::QPCompressedPolynomial, X, theta::Number, data::Vector)
    return vec(Eval(M, X, InterpolationWeights(M, [theta]), view(reshape(data, :, 1), M.perp_idx, :), reshape(data, :, 1)))
end

function L0_DF_parts(M::QPCompressedPolynomial, X, theta, data, L0, sel; cache=makeCache(M, X, theta, data))
    DD = L0_DF_parts(M.M.manifolds[sel[1]], X.x[sel[1]], theta, view(data, M.perp_idx, :), data, L0=L0, ii=sel[2], DV=cache.x[sel[1]])
    return DD
end

function L0_DF(M::QPCompressedPolynomial, X, theta, data, L0; cache=makeCache(M, X, theta, view(data, M.perp_idx, :), data))
    return ArrayPartition(map((m, x, c) -> L0_DF(m, x, theta, view(data, M.perp_idx, :), data, L0=L0, DV=c), M.M.manifolds, X.x, cache.x))
end

function DF_dt_parts(M::QPCompressedPolynomial, X, theta, data, sel; dt, cache=makeCache(M, X, theta, view(data, M.perp_idx, :), data))
    DF_dt_parts(M.M.manifolds[sel[1]], X.x[sel[1]], theta, view(data, M.perp_idx, :), data; dt=dt, ii=sel[2], DV=cache.x[sel[1]])
end
