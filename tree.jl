"""
Functions and stucture needed to describe a tree, this is used to create
an object to supplement the optimal classification tree.
"""
module tf
    struct Tree
        nodes::Array{Int64,1}
        branches::Array{Int64,1}
        leaves::Array{Int64,1}
        a::Dict{Int64,Int64}
        b::Dict{Int64,Float64}
        z::Dict{Int64,Int64}
    end

    using Random, DecisionTree, LinearAlgebra

    @inline get_e(p::Int) = 1*Matrix(I,p,p)
    @inline N_nodes(D::Int) = 2^(D+1) - 1
    @inline N_branch(D::Int) = Int(floor(N_nodes(D::Int)/2))
    @inline get_parent(node::Int) = Int(floor(node/2))

    function warm_start(depth::Int,y::Array{String,1},x::Array{Float64,2},seed::Int; rf_ntrees = 1)
        ```
        Obtain warm start for a local search tree

        Parameters:
        depth - starting depth of tree
        y - response y
        x - covariates variables
        seed - randomised seed
        ```
        n = size(x,1) #num observations
        p = size(x,2) #num features
        T = Tree([],[],[],Dict(),Dict(),Dict())
        e = get_e(p)
        #warmstart with randomForest
        seed = Random.seed!(seed)
        rf = build_forest(y,x,floor(Int,sqrt(p)),rf_ntrees,0.7,depth,5,2,rng = seed)
        obj = rf.trees[1]
        T = _branch_constraint(obj,1,T,e) #get the branching constraints from the randomForest
        T = assign_class(x,T,e) #fill the z matrix with classses
        return T
    end

    @inline function get_OCT(depth::Int)
        nodes = collect(1:N_nodes(depth))
        branches = collect(1:N_branch(depth))
        leaves = collect(N_branch(depth)+1:N_nodes(depth))
        return Tree(nodes,branches,leaves,Dict(),Dict(),Dict())
    end


    function get_children(node::Int,T::Tree)
        ```
        Get children of current node, calls recursive function to get all kids

        Parameters:
        node - any input node on tree
        T - tree
        ```
        kids = _recurse_children(node::Int,T)
        @inbounds if length(kids) == 1
            return
        else
            return kids
        end
    end


    function _recurse_children(node::Int,T::Tree)
        ```
        Recursive function to obtain tree until leaves are reached
        ```
        #cascade observation down the split
        kids = []
        @inbounds if node in T.branches
            append!(kids,left_child(node::Int,T))
            append!(kids,_recurse_children(left_child(node::Int,T),T))
            append!(kids,right_child(node::Int,T))
            append!(kids,_recurse_children(right_child(node::Int,T),T))
        else
            append!(kids,node)
        end
        return(unique(kids)) # TODO FIX LATER
    end

    function _branch_constraint(obj,node::Int,t::Tree,e::Array{Int64,2})
        ```
        Branching constraint used only on the warm start.

        obj -
        t - tree
        e - identity matrix
        ```
        append!(t.nodes,node)
        @inbounds if typeof(obj) != Leaf{String} #if the node is a branch get the branching constraints
            t.a[node] = obj.featid
            t.b[node] = obj.featval
            append!(t.branches,node)
            _branch_constraint(obj.left,2*node,t,e)
            _branch_constraint(obj.right,2*node+1,t,e)
        else
            append!(t.leaves,node)
        end
        return t
    end

<<<<<<< HEAD
    function get_thread_idx(num_threads, observations)
        if num_threads < size(observations,1)
            subtrees_per_thread = Int(floor(size(observations,1)/num_threads))
            indices = [subtrees_per_thread*t for t =1:num_threads]
            if mod(size(observations,1),num_threads) != 0
                n_remaining = size(observations,1) - subtrees_per_thread*num_threads
                indices_expand  = collect(1:n_remaining)
                filled_list = fill(indices_expand[end], num_threads- n_remaining)
                delta = [indices_expand;filled_list]
                output = [0;indices .+ delta]
            else
                output = [0;indices]
            end
            return output
        elseif num_threads >= size(observations,1)
            return [0;collect(1:size(observations,1))]
        end
    end

    function assign_class(X,T,e;indices = false)
    ```
    Assign all z values based on a and b branching values
    Inputs - Any tree with branching constraints
    Outputs - Tree with zs filled.
    ```
    println("assign CALSS SERIAL")
    if indices == false
        N = 1:size(X,1)
    else
        N = indices
    end
    #for each observation, find leaf and assign class
    for i in N
        # node = 1
        node = minimum(T.nodes)
        _cascade_down(node,X,i,T,e)
    end
    return T
end

    function assign_class(X,T,e,numthreads;indices = false)
=======
    function assign_class(X::Array{Float64,2},T::Tree,e::Array{Int64,2};indices = false)
>>>>>>> fa8f576780f544d388989e0a3323cb09fa843e47
        ```
        Assign all z values based on a and b branching values

        Inputs - Any tree with branching constraints
        Outputs - Tree with zs filled.
        ```
        @inbounds if indices == false
            N = 1:size(X,1)
        else
            N = indices
        end
        # println("ASSIGN CLASS PARALLALE")

        node = minimum(T.nodes)
        veccc = Int.([1:size(X,1) zeros(size(X,1))])
        vars = @view veccc[1:end,:]
        idx = tf.get_thread_idx(numthreads, X)
        Threads.@threads for thread in 1:numthreads
            @inbounds begin
                if thread <= size(X,1)
                    for i in vars[:,1][1+idx[thread]:idx[thread+1]]
                        # println("thread$thread")
                        vars[i,2] = _cascade_down_parallel(node,X,i,T,e)
                    end
                end
            end
        end

        output_z = Dict()
        for (k,v) in zip(veccc[:,1],veccc[:,2])
            output_z[k] = v
        end

        output_T = tf.Tree(T.nodes,T.branches,T.leaves,T.a,T.b,output_z)


        return output_T
    end

    function _cascade_down_parallel(node::Int,X,i::Int,T,e)
        #cascade observation down the split
        if node in T.leaves
            return node
        else
            j = T.a[node]
            if isempty(j)
                _cascade_down_parallel(tf.right_child(node,T),X,i,T,e)
            else
                if (e[:,j[1]]'*X[i,:] < T.b[node])
                    _cascade_down_parallel(tf.left_child(node,T),X,i,T,e)
                else
                    _cascade_down_parallel(tf.right_child(node,T),X,i,T,e)
                end
            end
        end
    end

    function _cascade_down(node::Int,X,i::Int,T::Tree,e::Array{Int64,2})
        #cascade observation down the split
        @inbounds if node in T.leaves
            T.z[i] = node
        else
            j = T.a[node]
            if isempty(j)
                _cascade_down(right_child(node,T),X,i,T,e)
            else
                if (e[:,j[1]]'*X[i,:] < T.b[node])
                    _cascade_down(left_child(node,T),X,i,T,e)
                else
                    _cascade_down(right_child(node,T),X,i,T,e)
                end
            end
        end
        return(T)
    end


    @inline function left_child(node::Int,T::Tree)#::Int,T::Tree)
        #returns left child node if exists in current tree
        @inbounds if 2*node in T.nodes
            return(2*node)
        end
    end

    @inline function right_child(node::Int,T::Tree)#::Int,T::Tree)
        #returns right child node if exists in current tree
        @inbounds if 2*node+1 in T.nodes
            return(2*node+1)
        end
    end

    @inline function minleafsize(T::Tree)
        # Get minimmum leaf size across all leaves in the tree
        minbucket = Inf
        Nt = zeros(maximum(T.nodes))
        z_mat = zeros(maximum([k for (k,v) in T.z]),maximum(T.nodes))
        @inbounds for t in T.leaves
            i = [k for (k,v) in T.z if v==t]
            z_mat[i,t] .= 1
            Nt[t] = sum(z_mat[:,t])
            if (Nt[t] < minbucket)
                minbucket = Nt[t]
            end
        end

        return(minbucket)
    end

    @inline function R(node::Int)
        # Get Right ancestors
        right_ancestors = []
        @inbounds if node==1
            return()
        elseif (node-1)/2==get_parent(node)
            append!(right_ancestors,get_parent(node))
            append!(right_ancestors,R(get_parent(node)))
        else
            append!(right_ancestors,R(get_parent(node)))
        end
        return(right_ancestors)
    end

    @inline function L(node::Int)
        # Get left ancestors
        left_ancestors = []
        @inbounds if node==1
            return()
        elseif node/2==get_parent(node)
            append!(left_ancestors,get_parent(node))
            append!(left_ancestors,L(get_parent(node)))
        else
            append!(left_ancestors,L(get_parent(node)))
        end
        return(left_ancestors)
    end

    function copy(Told::Tree)
        # Deep copy of a tree
        Tnew = Tree(deepcopy(Told.nodes),deepcopy(Told.branches),
                    deepcopy(Told.leaves),deepcopy(Told.a),
                    deepcopy(Told.b),deepcopy(Told.z))
        return Tnew
    end

    @inline function _nodes_subtree(node::Int,t::Tree)#::Int,t::Tree)
        # Get Subtree of a specified node
        subtree_nodes = []
        subtree_leaves = []
        @inbounds if node in t.leaves
            append!(subtree_nodes,node)
            append!(subtree_leaves,node)
        else
            append!(subtree_nodes,node)
            append!(subtree_nodes,_nodes_subtree(tf.left_child(node,t),t))
            append!(subtree_nodes,_nodes_subtree(tf.right_child(node,t),t))
        end
        return(subtree_nodes)
    end

    function create_subtree(node::Int,t::Tree)#,t::Tree)
        ```
        Return a subtree for a tree that has the node on the branch
        ```
        nodes = _nodes_subtree(node,t)
        #return new struct containing subtree
        leaves = t.leaves[(in(nodes).(t.leaves))]
        branches = nodes[(.!(in(leaves).(nodes)))]
        # println("branches", branches)
        @inbounds bs = Dict(key => t.b[key] for key in branches)
        @inbounds as = Dict(key => t.a[key] for key in branches)
        @inbounds zs = Dict(k => t.z[k] for (k,v) in t.z if v in leaves)
        return Tree(nodes,branches,leaves,as,bs,zs)
    end


    @inline function y_mat(y::Array{String,1})
        ```
        Get y matrix as a matrix of categorical
        ```
        y2 = deepcopy(y)
        Y = zeros(length(y2),length(unique(y2)))
        @inbounds for (i,x) in enumerate(unique(y2))
            Y[:,i] = (y.==x)*1
        end
        return Y
    end


    @inline function subtree_inputs(Tt::Tree,x::Array{Float64,2},y::Array{String,1})
        #returns a list of indices of observations contained in leaf nodes of subtree
        Y = y_mat(y)
        obs = Vector{Int64}()
        @inbounds for leaf in Tt.leaves
            append!(obs,[k for (k,v) in Tt.z if v==leaf])
        end
        return(obs)#,x[obs,:],Y[obs,:])
    end

end
