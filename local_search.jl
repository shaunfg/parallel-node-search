using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
include("tree.jl")
include("local_search_deep.jl")


# cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
# cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")
# include("tree.jl")

function splitobs(x::Array{Float64,2},y::Array{String,1},pct_train::Float64)
    xtrain = x[1:Int(floor(pct_train*size(x)[1])),:]
    xvalid = x[Int(floor(pct_train*size(x)[1]))+1:end,:]
    ytrain = y[1:Int(floor(pct_train*size(x)[1]))]
    yvalid = y[Int(floor(pct_train*size(x)[1]))+1:end]
    return(xtrain,xvalid,ytrain,yvalid)
end


function threaded_restarts!(x::Array{Float64,2},y::Array{String,1},
    nrestarts::Int64,tdepth::Int64,seed_values::Vector{Int64},
    n_threads::Int;tol_limit = 1e-3,α=0.001,warmup=400)
    #seed_values = 100:100:100*nrestarts
    threads_idx = tf.get_thread_idx(n_threads,seed_values)
    output_tree =  Dict()
    # output_tree = Vector{tf.Tree}(undef,length(seed_values))
    # v = @view output_tree[1:end,:]
    # seed = @view seed_values[1:end,:]
    # println(threads_idx)
    @inbounds Threads.@threads for t in 1:n_threads
        if t < length(threads_idx)
            for i = threads_idx[t]+1:threads_idx[t+1]
                # v[i] = LocalSearch(x,y,tdepth,seed[i],tol_limit=tol_limit,α=α)
                output_tree[i] = LocalSearch(x,y,tdepth,seed_values[i],tol_limit=tol_limit,α=α)
            end
        end
    end
    return(output_tree)
end

function serial_restarts!(x::Array{Float64,2},y::Array{String,1},nrestarts::Int64,
    tdepth::Int64;tol_limit = 1e-3,α=0.001,warmup=400)
    seed_values = 100:100:100*nrestarts
    output_tree = Dict()
    #we need to perform the calculations separately on remaining columns when there are remainder columns
    indices = 1:nrestarts
    @inbounds for j in indices
        seed = seed_values[j]
        #println(seed)
        output_tree[j] = LocalSearch(x,y,tdepth,seed,tol_limit=tol_limit,α=α)
    end
    return(output_tree)
end

function LocalSearch(x::Array{Float64,2},y::Array{String,1},tdepth::Int,seed::Int;
    tol_limit = 1e-3,α=0.001,deep =false)
    # println("##############################")
    # println("### Local Search Algorithm ###")
    # println("##############################")

    global Y = tf.y_mat(y)

    # Data pre-processing - Normalisation
    dt = fit(UnitRangeTransform, x, dims=1)
    X = StatsBase.transform(dt,x)
    e = tf.get_e(size(x,2))
    local T = tf.warm_start(tdepth,y,X,seed)

    # test_tree(T)
    starting_loss = loss(T,α)
    tol = 10
    local iter = 0
    @inbounds while tol > tol_limit
        iter +=1
        # println("Iteration # $iter")
        Lprev = loss(T,α)
        local Lcur
        local shuffled_t = shuffle(T.nodes)
        # print(shuffled_t)
        @inbounds for t in shuffled_t
            @inbounds if t ∈ T.nodes
                # println("(node $t)")
                # println("STARTING TREE node $t-- $T")
                Tt = tf.create_subtree(t,T)
                # test_tree(Tt)
                # println("STARTING node $t-- $Tt")
                local indices = tf.subtree_inputs(Tt,X,y)
                # println(length(indices))
                if deep == true
                    Ttnew, better_found = optimize_node_parallel_deep(Tt,indices,X,y,T,e,α)
                else
                    Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T,e;α=α)
                end
                if better_found ==true
                    T = replace_subtree(T,Ttnew,X;print_prog=false)
                    # global output = T
                    # println("replaced Tree $t-- $T")
                end
                Lcur = loss(T,α)
                tol = abs(Lprev - Lcur)
                #println("Lprev $Lprev, Lcur $Lcur")
            end
        end
        #println("$iter)Tolerance = $tol, Error = $Lcur, starting error = $starting_loss")
    end
    return T
end

@inline get_level(node::Int,subtree_root::Int) = Int(floor(log2(node/subtree_root)))
@inline calculate_destination(parent_root::Int,subtree_root::Int,node::Int) =
    node + (parent_root-subtree_root)*2^(get_level(node,subtree_root))

function replace_lower_upper(T_full::tf.Tree,subtree::tf.Tree,X::Array{Float64,2}; print_prog = false)#::Tree,subtree::Tree
    local T = tf.copy(T_full)
    if length(subtree.nodes) == 1 #
        kid = minimum(subtree.nodes)
        parent = tf.get_parent(kid)
        # if print_prog == true
        #     println("replacement - leaf $kid --> $parent")
        # end
        children = tf.get_children(parent,T)
        delete!(T.a,parent) # removed parent from branch
        delete!(T.b,parent)
        filter!(p -> p.first ∉ children, T.b)
        filter!(p -> p.first ∉ children, T.a)
        filter!(x->x ≠parent,T.branches) # remove parent from branches
        filter!(x->x ∉children,T.nodes) # remove kid from nodes
        filter!(x->x ∉children,T.branches) # remove kid from branches
        filter!(x->x ∉children,T.leaves) # remove kid from leaves
        append!(T.leaves,parent) # add parent to leaves
        points = [k for (k,v) in T.z if v in children]
        @inbounds for point in points # add assignments of z to parent
            T.z[point] = parent
        end
    else
        subtree_parent = minimum(subtree.nodes) #7
        tree_parent = tf.get_parent(subtree_parent)#minimum(T.nodes) #3
        children = tf.get_children(tree_parent,T) #3 onwards
        CD(node) = calculate_destination(tree_parent,subtree_parent,node)
        filtered_nodes = [children;tree_parent]
        # println("filtered",filtered_nodes)
        filter!(x->x ∉ filtered_nodes, T.nodes) # remove kid from nodes
        filter!(x->x ∉ filtered_nodes,T.branches) # remove kid from nodes
        filter!(x->x ∉ filtered_nodes,T.leaves) # remove kid from nodes

        new_nodes = [CD(node) for node in subtree.nodes]
        new_branches = [CD(node) for node in subtree.branches]
        new_leaves = [CD(node) for node in subtree.leaves]

        # println("get children",subtree.nodes)
        # println(calculate_destination(tree_parent,subtree_parent,subtree.nodes))
        append!(T.nodes, new_nodes)
        append!(T.branches, new_branches)
        append!(T.leaves, new_leaves)
        # println(T.nodes)
        # test_tree(T)
        extra_branches = [k for (k,v) in T.b if k ∈ children]
        filter!(p -> p.first ∉ extra_branches, T.a)
        filter!(p -> p.first ∉ extra_branches, T.b)

        @inbounds for key in keys(subtree.a)
            T.a[CD(key)] = subtree.a[key]
            T.b[CD(key)] = subtree.b[key]
        end

        e = tf.get_e(size(X,2))
        T = tf.assign_class(X,T,e)
        # test_tree(T)
    end
    return T
end

function replace_subtree(T_full::tf.Tree,subtree::tf.Tree,X::Array{Float64,2}; print_prog = false)#::Tree,subtree::Tree
    local T = tf.copy(T_full)

    parent =  minimum(subtree.nodes)
    children = tf.get_children(parent,T)

    if isnothing(children)==false # must have children on T tree
        filter!(x->x ∉children,T.nodes) # remove kid from nodes
        filter!(x->x ∉children,T.branches) # remove kid from branches
        filter!(x->x ∉children,T.leaves) # remove kid from leaves
        filter!(p -> p.first ∉ children, T.a)
        filter!(p -> p.first ∉ children, T.b)
    end
    # if length(subtree.nodes) ==1
    #     filter!(p -> p.first ∉ parent, T.a)
    #     filter!(p -> p.first ∉ parent, T.b)
    # end
    filter!(x->x ≠parent,T.leaves) # remove parent from branches
    filter!(x->x ≠parent,T.branches) # remove parent from branches
    filter!(x->x ≠parent,T.nodes) # remove parent from branches

    append!(T.leaves,subtree.leaves) # add parent to leaves
    append!(T.branches,subtree.branches) # add parent to leaves
    append!(T.nodes,subtree.nodes) # add parent to leaves
    @inbounds for node in keys(subtree.b)
        T.b[node] = subtree.b[node] # add parent to leaves
        T.a[node] = subtree.a[node] # add parent to leaves
    end
    @inbounds for point in keys(subtree.z) # add assignments of z to parent
        T.z[point] = subtree.z[point]
    end
    return(T)
end

function tune(dmax,xtrain,ytrain,xvalid,yvalid;nthreads = length(Sys.cpu_info()) -1 )
    Cp_vals = [0.001, 0.01, 0.1]#, 1, 10]
    # Cp_vals = [0.001, 0.01, 0.1, 1, 10]
    d_vals = collect(1:dmax)
    p = size(xvalid,2) #num features
    e = tf.get_e(p)
    grid = zeros(length(Cp_vals) * length(d_vals),4)
    local j=1
    for v in Cp_vals, u in d_vals
            grid[j,1] = u
            grid[j,2] = v
            j+=1
    end
    vars = @view grid[1:end,:]
    d_vals = Int.(grid[1:end,1])
    d = @view d_vals[1:end,:]
    seed =1000
    Tvalid = Dict()
    #segment = Int(floor(size(vars,1)/nthreads))
    #index = [[i*segment for i=0:nthreads-1] ; Int(floor(size(vars,1)))]
    idx = tf.get_thread_idx(nthreads, xtrain)
    Threads.@threads for thread in 1:nthreads#+1
    @inbounds begin
            #for i=index[t-1]+1:index[t]
            if thread <= size(xtrain,1)
                for i=1+idx[thread]:idx[thread+1]
                    T = LocalSearch(xtrain,ytrain,d[i],seed;tol_limit = 1,α=vars[i,2])
                    Tvalid[i]= tf.Tree(deepcopy(T.nodes),deepcopy(T.branches),
                                deepcopy(T.leaves),deepcopy(T.a),
                                deepcopy(T.b),deepcopy(Dict()))
                    Tvalid[i] = tf.assign_class(xvalid,Tvalid[i],e;indices = false)
                    vars[i,3] = loss(Tvalid[i],tf.y_mat(yvalid),vars[i,2])
                    vars[i,4] = Threads.threadid()
                end
            end
        end
    end
    #println(vars)
    best = vars[argmin(vars[:,3]),:]
    # return (best[1],best[2])
    return vars
end


# Y_full = tf.y_mat(y)
@inline function loss(T::tf.Tree,Y_full::Array{Float64,2},α::Float64)
    L̂ = _get_baseline(Y_full)
    z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
    Y = Y_full[z_keys,:] # Reorganize Y values to z order
    z_values = collect(values(T.z)) #
    Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ T.leaves]
    Nt = length.([[k for (k,v) in T.z if v ==t] for t ∈ T.leaves])
    Lt = [Nt[t] - maximum(Nkt[t]) for t=1:length(T.leaves)]
    L = sum(Lt)#/L̂ #### + α*Cp......
    C = length(T.branches)
    # println("L̂ = $L̂,L = $L, C = $(α*C)")
    # return (L) #so that it not so small for now
    return (L/L̂*1 + α*C) #so that it not so small for now
end


loss(T,α) = loss(T,Y,α)

# TODO Can reduce this as there are no more indices ... tecnically
@inline function _get_baseline(Y::Array{Float64,2})
    Nt = sum((Y),dims=1)
    error = size(Y,1) - maximum(Nt)
    return error
end


# Input: Subtree T to optimize, training data X,y
# Output: Subtree T with optimized parallel split at root
function optimize_node_parallel(Tt::tf.Tree,indices::Array{Int64,1},
            X::Array{Float64,2},y::Array{String,1},T::tf.Tree,e::Array{Int64,2};α::Float64)
    Y =tf.y_mat(y)
    XI = X[indices,:]
    YI = Y[indices,:]
    root = minimum(Tt.nodes)
    Tnew = Tt
    Tbest = Tt

    better_split_found = false

    error_best = loss(Tt,α)
    #println("(Node $root)")
    if root in Tt.branches
        # println("Optimize-Node-Parallel : Branch split")
        # println("Tt     $Tt")
        Tnew = tf.create_subtree(root,Tt)
        #println("Timing Marker 5")
        # println("New tree $Tnew")
        Tlower_sub = tf.create_subtree(tf.left_child(root,Tt),Tt)
        Tupper_sub = tf.create_subtree(tf.right_child(root,Tt),Tt)
        #println("Timing Marker 4")
    else
        # println("Optimize-Node-Parallel : Leaf split")
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5,α=α)
        # println("Tnew",Tnew)
        if Tnew == false
            return (Tt,false)
        end
        Tlower_sub = tf.create_subtree(tf.left_child(root,Tnew),Tnew)
        Tupper_sub = tf.create_subtree(tf.right_child(root,Tnew),Tnew)
    end
    #println("Timing Marker 1")
    Tlower = replace_lower_upper(Tnew,Tlower_sub,X)
    #println("Timing Marker 2")
    Tupper = replace_lower_upper(Tnew,Tupper_sub,X)
    Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y,α=α)

    # println("Para tree $Tpara")
    error_lower = loss(Tlower,α)
    error_upper = loss(Tupper,α)
    #println("Timing Marker 3")

    if error_para < error_best
        # println("!! Better Split Found : subtree")
        # println("-->Error : $error_best => $error_para")
        Tt,error_best = Tpara,error_para
        better_split_found = true
    end

    if error_lower < error_best
        # println("!! Better Split Found : lower")
        # println("-->Error : $error_best => $error_lower")
        Tt,error_best = Tlower,error_lower
        better_split_found = true
    end

    if error_upper < error_best
        # println("!! Better Split Found : upper")
        # println("-->Error : $error_best => $error_upper")
        Tt,error_best = Tupper,error_upper
        better_split_found = true

    end
    return(Tt,better_split_found)
end

function new_feasiblesplit(root::Int64,XI::Array{Float64,2},YI::Array{Float64,2},
                Tt::tf.Tree,e::Array{Int64,2},indices::Array{Int,1},X::Array{Float64,2}
                ;Nmin=5,α::Float64)
    branches = [root]
    leaves = [2*root,2*root+1]
    nodes = [root,2*root,2*root+1]
    n,p = size(XI)
    Tfeas = tf.Tree(nodes,branches,leaves,Dict(),Dict(),Dict())

    new_values, error_best = _get_split(root,XI,YI,Tfeas,e,indices,X,y;Nmin=5,α=α)
    # println("newvalues",new_values)
    if new_values == false # no feasible split found
        return false
    else
        Tpara = tf.copy(Tfeas)
        Tpara.a[root] = Int(new_values[1])
        Tpara.b[root] = new_values[2]
        filter!(x->x≠root, Tpara.leaves)
        # println("TFEASIBLE $Tpara",indices)
        Tpara = tf.assign_class(X,Tpara,e;indices = indices)
    end
    return Tpara
end

function best_parallelsplit(root::Int64,XI::Array{Float64,2},YI::Array{Float64,2},
                Tt::tf.Tree,e::Array{Int,2},indices::Array{Int,1},X::Array{Float64,2},
                y::Array{String,1}
                ;Nmin=5,α::Float64)
    #println("test- in parallel split")
    new_values, error_best = _get_split(root,XI,YI,Tt,e,indices,X,y;Nmin=5,α)
    if new_values == false # no feasible split found
        return (Tt, loss(Tt,α))
    else
        Tpara = tf.copy(Tt)
        Tpara.a[root] = Int(new_values[1])
        Tpara.b[root] = new_values[2]
        filter!(x->x≠root, Tpara.leaves)
        return (Tpara,error_best)
    end
end

function _get_split(root::Int64,XI::Array{Float64,2},YI::Array{Float64,2},
                Tt::tf.Tree,e::Array{Int64,2},indices::Array{Int,1},X::Array{Float64,2},
                y::Array{String,1};Nmin=5,α::Float64)
    n,p = size(XI)
    error_best = Inf
    Tttry = tf.copy(Tt)
    filter!(x->x≠root, Tttry.leaves)
    better_split_found = false
    # local new_values = [1,0.5]
    # local error_best = loss(Tt,y)
    local new_values, error_best
    @inbounds for j in 1:p, i in 1:n-1
        values = sort(XI[:,j])
        bsplit = 0.5*(values[i] + values[i+1])
        Tttry.a[root] = j
        Tttry.b[root] = bsplit
        Tttry = tf.assign_class(X,Tttry,e;indices = indices)
        #create a tree with this new a and b
        error = loss(Tttry,α)
        # println("MIN LEAF SIZE",tf.minleafsize(Tttry))
        if tf.minleafsize(Tttry) >= Nmin && error < error_best && true == true
            # println("MIN LEAF SIZE",tf.minleafsize(Tttry))
            error_best = error
            new_values = [j,bsplit]
            # println("newvalues before",new_values,indices)
            better_split_found = true
        end
    end
    if better_split_found == true
        # println("newvalues after",new_values)
        # println("FOUND FEASIBLE TREE")
        return(new_values,error_best)
    else
        #println("NO FEASIBLE TREE")
        return(false,false)
    end
end

#
#
#
#
# function LocalSearch_z(x::Array{Float64,2},y::Array{String,1},tdepth::Int,seed::Int;
#     tol_limit = 1e-3,α=0.000001,numthreads=1)
#
#     global Y = tf.y_mat(y)
#
#     # Data pre-processing - Normalisation
#     dt = fit(UnitRangeTransform, x, dims=1)
#     X = StatsBase.transform(dt,x)
#     e = tf.get_e(size(x,2))
#     local T = tf.warm_start(tdepth,y,X,seed)
#
#     starting_loss = loss(T,α)
#     tol = 10
#     local iter = 0
#     while tol > tol_limit
#         iter +=1
#         Lprev = loss(T,α)
#         local Lcur
#         local shuffled_t = shuffle(T.nodes)
#         # print(shuffled_t)
#         @inbounds for t in shuffled_t
#             @inbounds if t ∈ T.nodes
#                 # println("STARTING TREE node $t-- $T")
#                 Tt = tf.create_subtree(t,T)
#                 # println("STARTING node $t-- $Tt")
#                 local indices = tf.subtree_inputs(Tt,X,y)
#                 # println(length(indices))
#                 Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T,e;α=α,numthreads=numthreads)
#                 if better_found ==true
#                     T = replace_subtree(T,Ttnew,X;print_prog=true)
#                     # global output = T
#                     # println("replaced Tree $t-- $T")
#                 end
#                 Lcur = loss(T,α)
#                 tol = abs(Lprev - Lcur)
#                 println("Lprev $Lprev, Lcur $Lcur")
#             end
#         end
#         println("$iter)Tolerance = $tol, Error = $Lcur, starting error = $starting_loss")
#     end
#     return T
# end
#
# function replace_lower_upper(T_full::tf.Tree,subtree::tf.Tree,X::Array{Float64,2},
#     numthreads::Int64; print_prog = false)#::Tree,subtree::Tree
#     local T = tf.copy(T_full)
#     if length(subtree.nodes) == 1 #
#         kid = minimum(subtree.nodes)
#         parent = tf.get_parent(kid)
#         # if print_prog == true
#         #     println("replacement - leaf $kid --> $parent")
#         # end
#         children = tf.get_children(parent,T)
#         delete!(T.a,parent) # removed parent from branch
#         delete!(T.b,parent)
#         filter!(p -> p.first ∉ children, T.b)
#         filter!(p -> p.first ∉ children, T.a)
#         filter!(x->x ≠parent,T.branches) # remove parent from branches
#         filter!(x->x ∉children,T.nodes) # remove kid from nodes
#         filter!(x->x ∉children,T.branches) # remove kid from branches
#         filter!(x->x ∉children,T.leaves) # remove kid from leaves
#         append!(T.leaves,parent) # add parent to leaves
#         points = [k for (k,v) in T.z if v in children]
#         @inbounds for point in points # add assignments of z to parent
#             T.z[point] = parent
#         end
#     else
#         subtree_parent = minimum(subtree.nodes) #7
#         tree_parent = tf.get_parent(subtree_parent)#minimum(T.nodes) #3
#         children = tf.get_children(tree_parent,T) #3 onwards
#         CD(node) = calculate_destination(tree_parent,subtree_parent,node)
#         filtered_nodes = [children;tree_parent]
#         # println("filtered",filtered_nodes)
#         filter!(x->x ∉ filtered_nodes, T.nodes) # remove kid from nodes
#         filter!(x->x ∉ filtered_nodes,T.branches) # remove kid from nodes
#         filter!(x->x ∉ filtered_nodes,T.leaves) # remove kid from nodes
#
#         new_nodes = [CD(node) for node in subtree.nodes]
#         new_branches = [CD(node) for node in subtree.branches]
#         new_leaves = [CD(node) for node in subtree.leaves]
#
#         # println("get children",subtree.nodes)
#         # println(calculate_destination(tree_parent,subtree_parent,subtree.nodes))
#         append!(T.nodes, new_nodes)
#         append!(T.branches, new_branches)
#         append!(T.leaves, new_leaves)
#         # println(T.nodes)
#         # test_tree(T)
#         extra_branches = [k for (k,v) in T.b if k ∈ children]
#         filter!(p -> p.first ∉ extra_branches, T.a)
#         filter!(p -> p.first ∉ extra_branches, T.b)
#
#         @inbounds for key in keys(subtree.a)
#             T.a[CD(key)] = subtree.a[key]
#             T.b[CD(key)] = subtree.b[key]
#         end
#
#         e = tf.get_e(size(X,2))
#         T = tf.assign_class(X,T,e,numthreads)
#         # test_tree(T)
#     end
#     return T
# end
#
# # Input: Subtree T to optimize, training data X,y
# # Output: Subtree T with optimized parallel split at root
# function optimize_node_parallel(Tt::tf.Tree,indices::Array{Int64,1},
#             X::Array{Float64,2},y::Array{String,1},T::tf.Tree,e::Array{Int64,2},
#             numthreads::Int64;α::Float64)
#     Y =tf.y_mat(y)
#     XI = X[indices,:]
#     YI = Y[indices,:]
#     root = minimum(Tt.nodes)
#     Tnew = Tt
#     Tbest = Tt
#
#     better_split_found = false
#
#     error_best = loss(Tt,α)
#     #println("(Node $root)")
#     if root in Tt.branches
#         # println("Optimize-Node-Parallel : Branch split")
#         # println("Tt     $Tt")
#         Tnew = tf.create_subtree(root,Tt)
#         #println("Timing Marker 5")
#         # println("New tree $Tnew")
#         Tlower_sub = tf.create_subtree(tf.left_child(root,Tt),Tt)
#         Tupper_sub = tf.create_subtree(tf.right_child(root,Tt),Tt)
#         #println("Timing Marker 4")
#     else
#         # println("Optimize-Node-Parallel : Leaf split")
#         Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X,numthreads;Nmin=5,α=α)
#         # println("Tnew",Tnew)
#         if Tnew == false
#             return (Tt,false)
#         end
#         Tlower_sub = tf.create_subtree(tf.left_child(root,Tnew),Tnew)
#         Tupper_sub = tf.create_subtree(tf.right_child(root,Tnew),Tnew)
#     end
#     #println("Timing Marker 1")
#     Tlower = replace_lower_upper(Tnew,Tlower_sub,X)
#     #println("Timing Marker 2")
#     Tupper = replace_lower_upper(Tnew,Tupper_sub,X)
#     Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y,numthreads,α=α)
#
#     # println("Para tree $Tpara")
#     error_lower = loss(Tlower,α)
#     error_upper = loss(Tupper,α)
#     #println("Timing Marker 3")
#
#     if error_para < error_best
#         # println("!! Better Split Found : subtree")
#         # println("-->Error : $error_best => $error_para")
#         Tt,error_best = Tpara,error_para
#         better_split_found = true
#     end
#
#     if error_lower < error_best
#         # println("!! Better Split Found : lower")
#         # println("-->Error : $error_best => $error_lower")
#         Tt,error_best = Tlower,error_lower
#         better_split_found = true
#     end
#
#     if error_upper < error_best
#         # println("!! Better Split Found : upper")
#         # println("-->Error : $error_best => $error_upper")
#         Tt,error_best = Tupper,error_upper
#         better_split_found = true
#
#     end
#     return(Tt,better_split_found)
# end
#
# function new_feasiblesplit(root::Int64,XI::Array{Float64,2},YI::Array{Float64,2},
#                 Tt::tf.Tree,e::Array{Int64,2},indices::Array{Int,1},X::Array{Float64,2},
#                 numthreads::Int64;Nmin=5,α::Float64)
#     branches = [root]
#     leaves = [2*root,2*root+1]
#     nodes = [root,2*root,2*root+1]
#     n,p = size(XI)
#     Tfeas = tf.Tree(nodes,branches,leaves,Dict(),Dict(),Dict())
#
#     new_values, error_best = _get_split(root,XI,YI,Tfeas,e,indices,X,y,numthreads;Nmin=5,α=α)
#     # println("newvalues",new_values)
#     if new_values == false # no feasible split found
#         return false
#     else
#         Tpara = tf.copy(Tfeas)
#         Tpara.a[root] = Int(new_values[1])
#         Tpara.b[root] = new_values[2]
#         filter!(x->x≠root, Tpara.leaves)
#         # println("TFEASIBLE $Tpara",indices)
#         Tpara = tf.assign_class(X,Tpara,e,numthreads;indices = indices)
#     end
#     return Tpara
# end
#
# function best_parallelsplit(root::Int64,XI::Array{Float64,2},YI::Array{Float64,2},
#                 Tt::tf.Tree,e::Array{Int,2},indices::Array{Int,1},X::Array{Float64,2},
#                 y::Array{String,1},numthreads::Int64;Nmin=5,α::Float64)
#     #println("test- in parallel split")
#     new_values, error_best = _get_split(root,XI,YI,Tt,e,indices,X,y,numthreads;Nmin=5,α)
#     if new_values == false # no feasible split found
#         return (Tt, loss(Tt,α))
#     else
#         Tpara = tf.copy(Tt)
#         Tpara.a[root] = Int(new_values[1])
#         Tpara.b[root] = new_values[2]
#         filter!(x->x≠root, Tpara.leaves)
#         return (Tpara,error_best)
#     end
# end
#
# function _get_split(root::Int64,XI::Array{Float64,2},YI::Array{Float64,2},
#                 Tt::tf.Tree,e::Array{Int64,2},indices::Array{Int,1},X::Array{Float64,2},
#                 y::Array{String,1},numthreads::Int64;Nmin=5,α::Float64)
#     n,p = size(XI)
#     error_best = Inf
#     Tttry = tf.copy(Tt)
#     filter!(x->x≠root, Tttry.leaves)
#     better_split_found = false
#     # local new_values = [1,0.5]
#     # local error_best = loss(Tt,y)
#     local new_values, error_best
#     @inbounds for j in 1:p, i in 1:n-1
#         values = sort(XI[:,j])
#         bsplit = 0.5*(values[i] + values[i+1])
#         Tttry.a[root] = j
#         Tttry.b[root] = bsplit
#         Tttry = tf.assign_class(X,Tttry,e,numthreads;indices = indices)
#         #create a tree with this new a and b
#         error = loss(Tttry,α)
#         # println("MIN LEAF SIZE",tf.minleafsize(Tttry))
#         if tf.minleafsize(Tttry) >= Nmin && error < error_best && true == true
#             # println("MIN LEAF SIZE",tf.minleafsize(Tttry))
#             error_best = error
#             new_values = [j,bsplit]
#             # println("newvalues before",new_values,indices)
#             better_split_found = true
#         end
#     end
#     if better_split_found == true
#         # println("newvalues after",new_values)
#         # println("FOUND FEASIBLE TREE")
#         return(new_values,error_best)
#     else
#         #println("NO FEASIBLE TREE")
#         return(false,false)
#     end
# end
