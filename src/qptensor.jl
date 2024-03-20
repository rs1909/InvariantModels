
## ---------------------------------------------------------------------------------------
## TensorManifold
## 
## ---------------------------------------------------------------------------------------

struct TensorManifold{𝔽} <: AbstractManifold{𝔽}
    ranks::Array{T,1} where {T<:Integer}
    children::Array{T,2} where {T<:Integer}
    dim2ind::Array{T,1} where {T<:Integer}
    M::ProductManifold
    R::ProductRetraction
    VT::ProductVectorTransport
end

function getRetraction(M::TensorManifold{𝔽}) where {𝔽}
    return M.R
end

# internal
"""
    number of nodes of a HT tensor
"""
function nr_nodes(children::Array{T,2}) where {T<:Integer}
    return size(children, 1)
end

function nr_nodes(hten::TensorManifold{𝔽}) where {𝔽}
    return nr_nodes(hten.children)
end

"""
    check if a node is a leaf
"""
function is_leaf(children::Array{T,2}, ii) where {T<:Integer}
    return prod(children[ii, :] .== 0)
end

function is_leaf(hten::TensorManifold{𝔽}, ii) where {𝔽}
    return is_leaf(hten.children, ii)
end

# import Base.size
# 
# function size(M::TensorManifold{𝔽}) where T
#     return Tuple(size(X.x[k],1) for k in M.dim2ind)
# end

"""
    define_tree(d, tree_type = :balanced)
    creates a tree structure from dimensions 'd'
"""
function define_tree(d, tree_type=:balanced)
    children = zeros(typeof(d), 2 * d - 1, 2)
    dims = [collect(1:d)]

    nr_nodes = 1
    ii = 1
    while ii <= nr_nodes
        if length(dims[ii]) == 1
            children[ii, :] = [0, 0]
        else
            ii_left = nr_nodes + 1
            ii_right = nr_nodes + 2
            nr_nodes = nr_nodes + 2
            push!(dims, [])
            push!(dims, [])

            children[ii, :] = [ii_left, ii_right]
            if tree_type == :first_separate && ii == 1
                dims[ii_left] = [dims[ii][1]]
                dims[ii_right] = dims[ii][2:end]
            elseif tree_type == :first_pair_separate && ii == 1 && d > 2
                dims[ii_left] = dims[ii][1:2]
                dims[ii_right] = dims[ii][3:end]
            elseif tree_type == :TT
                dims[ii_left] = [dims[ii][1]]
                dims[ii_right] = dims[ii][2:end]
            else
                dims[ii_left] = dims[ii][1:div(end, 2)]
                dims[ii_right] = dims[ii][div(end, 2)+1:end]
            end
        end
        ii += 1
    end
    ind_leaves = findall(children[:, 1] .== 0)
    pivot = [dims[k][1] for k in ind_leaves]
    dim2ind = zero(pivot)
    dim2ind[pivot] = ind_leaves
    return children, dim2ind
end

# a complicated constructor
function TensorManifold(dims::Array{T,1}, topdim::T, ranks::Array{T,1}, children, dim2ind, tree_type=:balanced; field::AbstractNumbers=ℝ) where {T<:Integer}
    M = []
    R = []
    VT = []
    for ii = 1:nr_nodes(children)
        if is_leaf(children, ii)
            dim_id = findfirst(isequal(ii), dim2ind)
            n_ii = dims[dim_id]
            @assert n_ii >= ranks[ii] "Mismatched ranks."
            push!(M, Stiefel(n_ii, ranks[ii]))
            push!(R, PolarRetraction())
            push!(VT, DifferentiatedRetractionVectorTransport(PolarRetraction()))
        else
            ii_left = children[ii, 1]
            ii_right = children[ii, 2]
            if ii == 1
                push!(M, Euclidean(ranks[ii_left] * ranks[ii_right], ranks[ii]))
                push!(R, ExponentialRetraction())
                push!(VT, ParallelTransport())
            else
                @assert ranks[ii_left] * ranks[ii_right] >= ranks[ii] "Mismatched ranks."
                push!(M, Stiefel(ranks[ii_left] * ranks[ii_right], ranks[ii]))
                push!(R, PolarRetraction())
                push!(VT, DifferentiatedRetractionVectorTransport(PolarRetraction()))
            end
        end
    end
    return TensorManifold{field}(ranks, children, dim2ind, ProductManifold(M...), ProductRetraction(R...), ProductVectorTransport(VT...))
end

# create a rank structure such that 'ratio' is the lost rank
function cascade_ranks(children, dim2ind, topdim, dims; node_ratio=1.0, leaf_ratio=min(1.0, 2 / minimum(dims)), max_rank=18)
    ranks = zero(children[:, 1])
    ranks[1] = topdim
    for ii = nr_nodes(children):-1:2
        if is_leaf(children, ii)
            ldim = dims[findfirst(isequal(ii), dim2ind)]
            ranks[ii] = round(ldim * leaf_ratio)
            n_ii = ldim
            if ranks[ii] > min(n_ii, max_rank)
                ranks[ii] = min(n_ii, max_rank)
            end
        else
            ii_left = children[ii, 1]
            ii_right = children[ii, 2]
            ranks[ii] = round(ranks[ii_left] * ranks[ii_right] * node_ratio)
            if ranks[ii] > min(ranks[ii_left] * ranks[ii_right], max_rank)
                ranks[ii] = min(ranks[ii_left] * ranks[ii_right], max_rank)
            end
        end
    end
    return ranks
end

#  the output rank cannot be greater than the input rank
function prune_ranks!(dims::Array{T,1}, topdim::T, ranks::Array{T,1}, children, dim2ind) where {T<:Integer}
    ranks[1] = topdim
    for ii = nr_nodes(children):-1:2
        if is_leaf(children, ii)
            n_ii = dims[findfirst(isequal(ii), dim2ind)]
            if ranks[ii] > n_ii
                ranks[ii] = n_ii
            end
        else
            ii_left = children[ii, 1]
            ii_right = children[ii, 2]
            if ranks[ii] > ranks[ii_left] * ranks[ii_right]
                ranks[ii] = ranks[ii_left] * ranks[ii_right]
            end
        end
    end
    return nothing
end

function MinimalTensorManifold(dims::Array{T,1}, topdim::T=1, tree_type=:balanced) where {T<:Integer}
    children, dim2ind = define_tree(length(dims), tree_type)
    nodes = nr_nodes(children)
    # create ranks at each node
    ranks = ones(Int, nodes)
    # the root node is singular
    prune_ranks!(dims, topdim, ranks, children, dim2ind)
    return TensorManifold(dims, topdim, ranks, children, dim2ind, tree_type)
end

"""
    TODO: documentation
"""
function HTTensor(dims::Array{T,1}, topdim::T=1, tree_type=:balanced; node_rank=4, max_rank=18) where {T<:Integer}
    children, dim2ind = define_tree(length(dims), tree_type)
    nodes = nr_nodes(children)
    # create ranks at each node
    ranks = cascade_ranks(children, dim2ind, topdim, dims, node_ratio=1.0, leaf_ratio=min(1.0, node_rank / minimum(dims)), max_rank=max_rank)
    @show ranks
    return TensorManifold(dims, topdim, ranks, children, dim2ind, tree_type)
end

function project!(M::TensorManifold{field}, Y, p, X) where {field}
    return project!(M.M, Y, p, X)
end

function randn(M::TensorManifold{field}) where {field}
    return ArrayPartition(map(randn, M.M.manifolds))
end

function zero(M::TensorManifold{field}) where {field}
    return ArrayPartition(map(zero, M.M.manifolds))
end

function zero_vector!(M::TensorManifold{field}, X, p) where {field}
    return zero_vector!(M.M, X, p)
end

function manifold_dimension(M::TensorManifold)
    return manifold_dimension(M.M)
end

function inner(M::TensorManifold, p, X, Y)
    return inner(M.M, p, X, Y)
end

function retract!(M::TensorManifold, q, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    return retract!(M.M, q, p, X, t, method)
end

function retract(M::TensorManifold, p, X, t::Number, method::AbstractRetractionMethod=M.R)
    return retract(M.M, p, X, t, method)
end

function vector_transport_to!(M::TensorManifold, Y, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to!(M.M, Y, p, X, q, method)
end

function vector_transport_to(M::TensorManifold, p, X, q, method::AbstractVectorTransportMethod=M.VT)
    return vector_transport_to(M.M, p, X, q, method)
end

function get_coordinates(M::TensorManifold, p, X, B::AbstractBasis)
    return get_coordinates(M.M, p, X, B)
end

function get_vector(M::TensorManifold, p, c, B::AbstractBasis)
    return get_vector(M.M, p, c, B)
end

function getel(hten::TensorManifold{field}, X, idx) where {field}
    # this goes from the top level to the lowest level
    vecs = [zeros(Float64, 0) for k = 1:size(hten.children, 1)]
    for ii = size(hten.children, 1):-1:1
        if is_leaf(hten, ii)
            sel = idx[findfirst(isequal(ii), hten.dim2ind)]
            vecs[ii] = X.x[ii][sel, :]
        else
            ii_left = hten.children[ii, 1]
            ii_right = hten.children[ii, 2]
            s_l = size(X.x[ii_left], 2)
            s_r = size(X.x[ii_right], 2)
            vecs[ii] = [transpose(vecs[ii_left]) * reshape(X.x[ii][:, k], s_l, s_r) * vecs[ii_right] for k = 1:size(X.x[ii], 2)]
        end
    end
    return vecs[1][idx[end]]
end

"""
    Calculates the Gramian matrices of the HT tensor
"""
function gramians(hten::TensorManifold{field}, X) where {field}
    gr = Array{Array{Float64,2},1}(undef, size(hten.children, 1))
    gr[1] = transpose(X.x[1]) * X.x[1]
    for ii = 1:size(hten.children, 1)
        if !is_leaf(hten, ii)
            # Child nodes
            ii_left = hten.children[ii, 1]
            ii_right = hten.children[ii, 2]
            s_l = size(X.x[ii_left], 2)
            s_r = size(X.x[ii_right], 2)

            #             % Calculate contractions < B{ii}, G{ii} o_1 B{ii} >_(1, 2) and _(1, 3)
            #             B_mod = ttm(conj(x.B{ii}), G{ii}, 3);
            #             @show size(X.x[ii]), size(gr[ii])
            #             @show size(reshape(X.x[ii], s_l, s_r,:)), size(reshape(gr[ii], size(gr[ii],1), size(gr[ii],2), 1))
            B_mod = reshape(X.x[ii], s_l, s_r, :) * gr[ii]
            #   G{ii_left } = ttt(conj(x.B{ii}), B_mod, [2 3], [2 3], 1, 1);
            #   G{ii_right} = ttt(conj(x.B{ii}), B_mod, [1 3], [1 3], 2, 2);
            gr[ii_left] = dropdims(sum(reshape(X.x[ii], s_l, 1, s_r, :) .* reshape(B_mod, 1, s_l, s_r, :), dims=(3, 4)), dims=(3, 4))
            gr[ii_right] = dropdims(sum(reshape(X.x[ii], s_l, 1, s_r, :) .* reshape(B_mod, s_l, s_r, 1, :), dims=(1, 4)), dims=(1, 4))
        end
    end
    return gr
end

# normally we don't need this
# orthogonalise the non-root nodes
"""
    This orthogonalises the non-root nodes of the HT tensor, by leaving the value unchanged.
    Normally, this method is not needed, because our algorithms keep these matrices orthogonal
"""
function orthog!(M::TensorManifold{field}, X) where {field}
    for ii = nr_nodes(M):-1:2
        if is_leaf(M, ii)
            F = qr(X.x[ii])
            X.x[ii] .= Array(F.Q)
            R = F.R
        else
            F = qr(X.x[ii])
            X.x[ii] .= Array(F.Q)
            R = F.R
        end
        left_par = findfirst(isequal(ii), M.children[:, 1])
        right_par = findfirst(isequal(ii), M.children[:, 2])
        if left_par != nothing
            for k = 1:size(X.x[left_par], 2)
                X.x[left_par][:, k] .= vec(R * reshape(X.x[left_par][:, k], size(R, 2), :))
            end
        else
            for k = 1:size(X.x[right_par], 2)
                X.x[right_par][:, k] .= vec(reshape(X.x[right_par][:, k], :, size(R, 2)) * transpose(R))
            end
        end
    end
    return nothing
end

# -------------------------------------------------------------------------------------
#
# In this section we calculate the hmtensor derivative of X with respect to X.U, X.B
# The data is an array for multiple evaluations at the same time
# we need to allow for a missing index, so that a vector is output
# therefore we need to propagate a matrix through
#
# -------------------------------------------------------------------------------------

# this acts as a cache for the tensor evaluations and gradients
struct diadicVectors{T}
    valid_vecs::Array{Bool,1}
    valid_bvecs::Array{Bool,1}
    vecs::Array{Array{T,3},1}
    bvecs::Array{Array{T,3},1}
end

function invalidateVecs(DV::diadicVectors)
    DV.valid_vecs .= false
    nothing
end

function invalidateBVecs(DV::diadicVectors)
    DV.valid_bvecs .= false
    nothing
end

function invalidateAll(DV::diadicVectors)
    DV.valid_vecs .= false
    DV.valid_bvecs .= false
    nothing
end

function diadicVectors(T, nodes)
    return diadicVectors{T}(zeros(Bool, nodes), zeros(Bool, nodes), Array{Array{T,3},1}(undef, nodes), Array{Array{T,3},1}(undef, nodes))
end

function LmulBmulR!(vecs_ii, vecs_left, B, vecs_right)
    #     println("LmulBmulR!")
    #     @time for k=1:size(vecs_right,3)
    #         for jr = 1:size(vecs_right, 2)
    #             for jl = 1:size(vecs_left, 2)
    #                 for jb=1:size(B,2)
    #                     for q=1:size(vecs_right, 1)
    #                         for p=1:size(vecs_left, 1)
    #                             @inbounds vecs_ii[jb, jr*jl, k] += vecs_left[p,jl,k] * B[p + (q-1)*size(vecs_left, 1), jb] .* vecs_right[q,jr,k]
    #                         end
    #                     end
    #                 end
    #             end
    #         end
    #     end
    C = reshape(B, size(vecs_left, 1), size(vecs_right, 1), :)
    if size(vecs_left, 2) == 1
        @tullio vecs_ii[jb, jr, k] = vecs_left[p, 1, k] * C[p, q, jb] * vecs_right[q, jr, k]
    elseif size(vecs_right, 2) == 1
        @tullio vecs_ii[jb, jl, k] = vecs_left[p, jl, k] * C[p, q, jb] * vecs_right[q, 1, k]
    end
    return nothing
end

function BmulData!(vecs_ii, B, data_wdim)
    #     println("BmulData!")
    #     @time for k=1:size(data_wdim,2)
    #         for p=1:size(B,2)
    #             for q=1:size(B,1)
    #                 vecs_ii[p,1,k] += B[q,p] * data_wdim[q,k]
    #             end
    #         end
    #     end
    #     @show size(B), size(data_wdim)
    @tullio vecs_ii[p, 1, k] = B[q, p] * data_wdim[q, k]
    return nothing
end

# contracts data[i] with index[i] of the tensor X
# if i > length(data) data[end] is contracted with index[i]
# if third != 0, index[third] is not contracted
# output:
#   vecs is a matrix for each data point
# input:
#   X: is the tensor
#   data: is a vector of two dimensional arrays. 
#       The first dimension is the vector, the second dimension is the data index
#   ii: the node to contract at
#   second: if non-zero data[2] is contracted with this index, data[1] with the rest except 'third'
#   third: if non-zero, the index not to contract
#   dt: to replace X at index rep
#   rep: where to replace X with dt
function tensorVecsRecursive!(DV::diadicVectors{T}, M::TensorManifold{field}, X, data...; ii) where {field,T}
    #     print("v=", ii, "/", length(X.x), "_")
    if DV.valid_vecs[ii]
        return nothing
    end
    B = X.x[ii]
    if is_leaf(M, ii)
        # it is a leaf
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(data)
            wdim = length(data)
        else
            wdim = dim
        end
        if ~isassigned(DV.vecs, ii)
            DV.vecs[ii] = zeros(T, size(B, 2), 1, size(data[1], 2)) # Array{T}(undef, (size(B,2), 1, size(data[1],2)))
        end
        #         @show size(DV.vecs[ii]), size(B), size(data[wdim])
        BmulData!(DV.vecs[ii], B, data[wdim])
    else
        # it is a node
        ii_left = M.children[ii, 1]
        ii_right = M.children[ii, 2]
        tensorVecsRecursive!(DV, M, X, data..., ii=ii_left)
        tensorVecsRecursive!(DV, M, X, data..., ii=ii_right)
        vs_l = size(DV.vecs[ii_left], 2)
        vs_r = size(DV.vecs[ii_right], 2)
        vs = max(vs_l, vs_r)
        if ~isassigned(DV.vecs, ii)
            DV.vecs[ii] = zeros(size(B, 2), vs, size(data[1], 2))
        end

        LmulBmulR!(DV.vecs[ii], DV.vecs[ii_left], B, DV.vecs[ii_right])
    end
    DV.valid_vecs[ii] = true
    return nothing
end

# invalidates vecs that are dependent on node "ii"
function tensorVecsInvalidate(DV::diadicVectors{T}, M::TensorManifold{field}, ii) where {field,T}
    DV.valid_vecs[ii] = false
    # not the root node
    if ii != 1
        # find parent. Everything has a parent!
        left = findfirst(isequal(ii), M.children[:, 1])
        right = findfirst(isequal(ii), M.children[:, 2])
        if left != nothing
            tensorVecsInvalidate(DV, M, left)
        end
        if right != nothing
            tensorVecsInvalidate(DV, M, right)
        end
    end
    return nothing
end

# create partial results at the end of each node or leaf when multiplied with 'data'
function tensorVecs(M::TensorManifold{field}, X, data...; second::Integer=0, third::Integer=0, dt::ArrayPartition=ArrayPartition(), rep::Integer=0) where {field}
    DV = diadicVectors(eltype(data[end]), size(M.children, 1))
    tensorVecsRecursive!(DV, M, X, data..., ii=1)
    return DV
end

# L0 is a multiplication from the left
function Eval(M::TensorManifold{field}, X, data...; DV::diadicVectors=tensorVecs(M, X, data...)) where {field}
    return dropdims(DV.vecs[1], dims=2)
end


function BVpmulBmulV!(bvecs_ii::AbstractArray{T,3}, B::AbstractArray{U,2}, vecs_sibling::AbstractArray{T,3}, bvecs_parent::AbstractArray{T,3}) where {T,U}
    C = reshape(B, size(bvecs_ii, 2), size(vecs_sibling, 1), :)
    @tullio bvecs_ii[l, p, k] = C[p, q, r] * bvecs_parent[l, r, k] * vecs_sibling[q, 1, k]
    return nothing
end

function VmulBmulBVp!(bvecs_ii::AbstractArray{T,3}, vecs_sibling::AbstractArray{T,3}, B::AbstractArray{U,2}, bvecs_parent::AbstractArray{T,3}) where {T,U}
    C = reshape(B, size(vecs_sibling, 1), size(bvecs_ii, 2), :)
    @tullio bvecs_ii[l, q, k] = C[p, q, r] * bvecs_parent[l, r, k] * vecs_sibling[p, 1, k]
    return nothing
end

# Only supports full contraction
# dt is used for second derivatives. dt replaces X at node [rep].
# This is the same as taking the derivative w.r.t. node [rep] and multiplying by dt[rep]
# L0 is used to contract with tensor output
function tensorBVecsIndexed!(DV::diadicVectors{T}, M::TensorManifold{field}, X; ii) where {field,T}
    #     print("b=", ii, "/", length(X.x), "_")
    # find the parent and multiply with the vecs from the other brancs and the becs fron the bottom
    if DV.valid_bvecs[ii]
        return nothing
    end
    datalen = size(DV.vecs[1], 3)
    if ii == 1
        DV.bvecs[ii] = zeros(T, size(X.x[ii], 2), size(X.x[ii], 2), datalen)
        for k = 1:size(X.x[ii], 2)
            DV.bvecs[ii][k, k, :] .= 1
        end
    else
        # find parent. Everything has a parent!
        left = findfirst(isequal(ii), M.children[:, 1])
        right = findfirst(isequal(ii), M.children[:, 2])
        if left != nothing
            # B[left,right,parent] <- parent = BVecs[parent], right = vecs[sibling]
            parent = left
            B = X.x[parent]
            s_l = size(X.x[M.children[parent, 1]], 2)
            s_r = size(X.x[M.children[parent, 2]], 2)
            sibling = M.children[left, 2] # right sibling
            tensorBVecsIndexed!(DV, M, X, ii=parent)
            if ~isassigned(DV.bvecs, ii)
                DV.bvecs[ii] = zeros(T, size(DV.bvecs[parent], 1), s_l, datalen)
            end
            BVpmulBmulV!(DV.bvecs[ii], B, DV.vecs[sibling], DV.bvecs[parent])
        end
        if right != nothing
            # B[left,right,parent] <- parent = BVecs[parent], right = vecs[sibling]
            parent = right
            B = X.x[parent]
            s_l = size(X.x[M.children[parent, 1]], 2)
            s_r = size(X.x[M.children[parent, 2]], 2)

            sibling = M.children[right, 1] # right sibling
            tensorBVecsIndexed!(DV, M, X, ii=parent)
            #             @show size(X.B[parent]), size(DV.bvecs[parent]), size(vec(vecs[sibling]))
            if ~isassigned(DV.bvecs, ii)
                DV.bvecs[ii] = zeros(T, size(DV.bvecs[parent], 1), s_r, datalen)
            end
            VmulBmulBVp!(DV.bvecs[ii], DV.vecs[sibling], B, DV.bvecs[parent])
        end
    end
    DV.valid_bvecs[ii] = true
    return nothing
end


# if ii > 0  -> all children nodes get invalidate
function tensorBVecsInvalidate(DV::diadicVectors{T}, M::TensorManifold{field}, ii) where {field,T}
    # the root is always the identity, no need to update
    if is_leaf(M, ii)
        # do nothing as it has no children
    else
        ii_left = M.children[ii, 1]
        ii_right = M.children[ii, 2]
        DV.valid_bvecs[ii_left] = false
        DV.valid_bvecs[ii_right] = false
        tensorBVecsInvalidate(DV, M, ii_left)
        tensorBVecsInvalidate(DV, M, ii_right)
    end
    return nothing
end

function tensorBVecs!(DV::diadicVectors, M::TensorManifold{field}, X) where {field}
    if size(DV.vecs[1], 2) != 1
        println("only supports full contraction")
        return nothing
    end
    datalen = size(DV.vecs[1], 3)
    for ii = 1:nr_nodes(M)
        tensorBVecsIndexed!(DV, M, X, ii=ii)
    end
    return nothing
end

function makeCache(M::TensorManifold, X, data...)
    DV = diadicVectors(eltype(data[end]), size(M.children, 1))
    tensorVecsRecursive!(DV, M, X, data...; ii=1)
    tensorBVecs!(DV, M, X)
    return DV
end

# update the content which is invalidated
function updateCache!(DV::diadicVectors, M::TensorManifold, X, data::Vararg{AbstractMatrix{T}}) where {T}
    DV.valid_vecs .= false
    DV.valid_bvecs .= false
    tensorVecsRecursive!(DV, M, X, data..., ii=1)
    tensorBVecs!(DV, M, X)
    return nothing
end

function updateCachePartial!(DV::diadicVectors, M::TensorManifold, X, data::Vararg{AbstractMatrix{T}}; ii) where {T}
    tensorVecsInvalidate(DV, M, ii)
    tensorBVecsInvalidate(DV, M, ii)
    # make sure to update all the marked components
    # Vecs can start at the root (ii = 1), hence it covers the full tree
    tensorVecsRecursive!(DV, M, X, data..., ii=1)
    # BVecs has to start from all the leaves to cover the whole tree
    tensorBVecs!(DV, M, X)
    return nothing
end

# same as wDF, except that it only applies to node ii
function L0_DF_parts(M::TensorManifold{field}, X, data::Vararg{AbstractMatrix{T}}; L0, ii::Integer=-1, DV=makeCache(M, X, data...)) where {field,T}
    #     t0 = time()
    if is_leaf(M, ii)
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(data)
            wdim = length(data)
        else
            wdim = dim
        end
        dd = data[wdim]
        bv = DV.bvecs[ii]
        @tullio XO[p, q] := L0[l, k+0] * dd[p, k] * bv[l, q, k]
    else
        # it is a node
        ch1 = DV.vecs[M.children[ii, 1]]
        ch2 = DV.vecs[M.children[ii, 2]]
        bv = DV.bvecs[ii]
        @tullio XOp[p, q, r] := L0[l, k+0] * ch1[p, 1, k] * ch2[q, 1, k] * bv[l, r, k]
        XO = reshape(XOp, :, size(XOp, 3))
    end
    #     t1 = time()
    #     println("\n -> L0_DF = ", 100*(t1-t0))
    return XO
end

function L0_DF(M::TensorManifold{field}, X, data::Vararg{AbstractMatrix{T}}; L0, DV=makeCache(M, X, data...)) where {field,T}
    #     @show Tuple(collect(1:length(M.M.manifolds)))
    # @show size(data[1]), size(L0), size(DV.bvecs)
    return ArrayPartition(map((x) -> L0_DF_parts(M, X, data..., L0=L0, ii=x, DV=DV), Tuple(collect(1:length(M.M.manifolds)))))
end

# L0 is a square matrix ...
# function L0_DF1_DF2_parts(M::TensorManifold{field}, X, L0, dataX, dataY; ii::Integer = -1, DVX = makeCache(M, X, dataX), DVY = makeCache(M, X, dataY)) where field
#     if is_leaf(M, ii)
#         @tullio XO[p1,q1,p2,q2] := L0[r1,r2,k] * dataX[p1,k] * DVX.bvecs[ii][r1,q1,k] * dataY[p2,k] * DVY.bvecs[ii][r2,q2,k]
#         return XO
#     else
#         # it is a node
# #         @show ii, M.children[ii,1], M.children[ii,2]
#         chX1 = DVX.vecs[M.children[ii,1]]
#         chX2 = DVX.vecs[M.children[ii,2]]
#         bvX = DVX.bvecs[ii]
#         chY1 = DVY.vecs[M.children[ii,1]]
#         chY2 = DVY.vecs[M.children[ii,2]]
#         bvY = DVY.bvecs[ii]
#         @tullio XOp[p1,q1,r1,p2,q2,r2] := L0[l1,l2,k] * chX1[p1,1,k] * chX2[q1,1,k] * bvX[l1,r1,k] * chY1[p2,1,k] * chY2[q2,1,k] * bvY[l2,r2,k]
#         XO = reshape(XOp, size(XOp,1)*size(XOp,2), size(XOp,3), size(XOp,4)*size(XOp,5), size(XOp,6))
#         return XO
#     end
# end

# instead of multiplying the gradient from the left, we are multiplying it from the right
# there is no contraction along the indices of data...
function DF_dt_parts(M::TensorManifold{field}, X, data...; dt, ii, DV=makeCache(M, X, data...)) where {field}
    if is_leaf(M, ii)
        dim = findfirst(isequal(ii), M.dim2ind)
        if dim > length(data)
            wdim = length(data)
        else
            wdim = dim
        end
        l_data = data[wdim]
        bv = DV.bvecs[ii]
        # should be made non-allocating
        XO = zeros(eltype(dt), size(bv, 1), size(bv, 3))
        @tullio XO[l, k] = bv[l, q, k] * l_data[p, k+0] * dt[p, q]
    else
        s_l = size(X.x[M.children[ii, 1]], 2)
        s_r = size(X.x[M.children[ii, 2]], 2)
        dtp = reshape(dt, s_l, s_r, :)
        ch1 = DV.vecs[M.children[ii, 1]]
        ch2 = DV.vecs[M.children[ii, 2]]
        bv = DV.bvecs[ii]
        # should be made non-allocating
        XO = zeros(eltype(dtp), size(bv, 1), size(bv, 3))
        @tullio XO[l, k] = ch1[p, 1, k] * ch2[q, 1, k] * bv[l, r, k+0] * dtp[p, q, r]
    end
    #     @show size(data), size(dt), size(XO)
    return XO
end

function DF_dt(M::TensorManifold{field}, X, data...; dt, DV=makeCache(M, X, data...)) where {field}
    return ArrayPartition(map((x, y) -> DF_dt_parts(M, X, data..., dt=y, ii=x, DV=DV), Tuple(collect(1:length(M.M.manifolds))), dt.x))
end

