"""
Functions and stucture needed to describe a tree, this is used to create
an object to supplement the optimal classification tree.
"""
module tf
    struct Tree
        nodes
        branches
        leaves
        a
        b
        z
    end

    using Random, DecisionTree, LinearAlgebra, CategoricalArrays
    import MLJBase.int

    get_e(p) = 1*Matrix(I,p,p)
    N_nodes(D::Int) = 2^(D+1) - 1
    N_branch(D::Int) = Int(floor(N_nodes(D::Int)/2))
    get_parent(node) = Int(floor(node/2))

    function warm_start(depth::Int,y,x,seed; rf_ntrees = 1)
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

    function get_children(node::Int,T)
        ```
        Get children of current node, calls recursive function to get all kids

        Parameters:
        node - any input node on tree
        T - tree
        ```
        kids = _recurse_children(node::Int,T)
        if length(kids) == 1
            return
        else
            return kids
        end
    end


    function _recurse_children(node::Int,T)
        ```
        Recursive function to obtain tree until leaves are reached
        ```
        #cascade observation down the split
        kids = []
        if node in T.branches
            append!(kids,left_child(node::Int,T))
            append!(kids,_recurse_children(left_child(node::Int,T),T))
            append!(kids,right_child(node::Int,T))
            append!(kids,_recurse_children(right_child(node::Int,T),T))
        else
            append!(kids,node)
        end
        return(unique(kids)) # TODO FIX LATER
    end

    function _branch_constraint(obj,node,t,e)
        ```
        Branching constraint used only on the warm start.

        obj -
        t - tree
        e - identity matrix
        ```
        append!(t.nodes,node)
        if typeof(obj) != Leaf{String} #if the node is a branch get the branching constraints
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

    function assign_class(X,T,e;indices = false)
        ```
        Assign all z values based on a and b branching values

        Inputs - Any tree with branching constraints
        Outputs - Tree with zs filled.
        ```
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

    function _cascade_down(node::Int,X,i::Int,T,e)
        #cascade observation down the split
        if node in T.leaves
            T.z[i] = node
        else
            j = T.a[node]
            if isempty(j)
                tf._cascade_down(tf.right_child(node,T),X,i,T,e)
            else
                if (e[:,j[1]]'*X[i,:] < T.b[node])
                    tf._cascade_down(tf.left_child(node,T),X,i,T,e)
                else
                    tf._cascade_down(tf.right_child(node,T),X,i,T,e)
                end
            end
        end
        return(T)
    end


    function left_child(node,T)#::Int,T::Tree)
        #returns left child node if exists in current tree
        if 2*node in T.nodes
            return(2*node)
        end
    end

    function right_child(node,T)#::Int,T::Tree)
        #returns right child node if exists in current tree
        if 2*node+1 in T.nodes
            return(2*node+1)
        end
    end

    function minleafsize(T)
        # Get minimmum leaf size across all leaves in the tree
        minbucket = Inf
        Nt = zeros(maximum(T.nodes))
        z_mat = zeros(maximum([k for (k,v) in T.z]),maximum(T.nodes))
        for t in T.leaves
            i = [k for (k,v) in T.z if v==t]
            z_mat[i,t] .= 1
            Nt[t] = sum(z_mat[:,t])
            if (Nt[t] < minbucket)
                minbucket = Nt[t]
            end
        end
        return(minbucket)
    end

    function R(node::Int)
        # Get Right ancestors
        right_ancestors = []
        if node==1
            return()
        elseif (node-1)/2==get_parent(node)
            append!(right_ancestors,get_parent(node))
            append!(right_ancestors,R(get_parent(node)))
        else
            append!(right_ancestors,R(get_parent(node)))
        end
        return(right_ancestors)
    end

    function L(node::Int)
        # Get left ancestors
        left_ancestors = []
        if node==1
            return()
        elseif node/2==get_parent(node)
            append!(left_ancestors,get_parent(node))
            append!(left_ancestors,L(get_parent(node)))
        else
            append!(left_ancestors,L(get_parent(node)))
        end
        return(left_ancestors)
    end

    function copy(Told)
        # Deep copy of a tree
        Tnew = Tree(deepcopy(Told.nodes),deepcopy(Told.branches),
                    deepcopy(Told.leaves),deepcopy(Told.a),
                    deepcopy(Told.b),deepcopy(Told.z))
        return Tnew
    end

    function _nodes_subtree(node,t)#::Int,t::Tree)
        # Get Subtree of a specified node
        subtree_nodes = []
        subtree_leaves = []
        if node in t.leaves
            append!(subtree_nodes,node)
            append!(subtree_leaves,node)
        else
            append!(subtree_nodes,node)
            append!(subtree_nodes,_nodes_subtree(tf.left_child(node,t),t))
            append!(subtree_nodes,_nodes_subtree(tf.right_child(node,t),t))
        end
        return(subtree_nodes)
    end

    function create_subtree(node,t)#,t::Tree)
        ```
        Return a subtree for a tree that has the node on the branch
        ```
        nodes = _nodes_subtree(node,t)
        #return new struct containing subtree
        leaves = t.leaves[(in(nodes).(t.leaves))]
        branches = nodes[(.!(in(leaves).(nodes)))]
        # println("branches", branches)
        bs = Dict(key => t.b[key] for key in branches)
        as = Dict(key => t.a[key] for key in branches)
        zs = Dict(k => t.z[k] for (k,v) in t.z if v in leaves)
        return Tree(nodes,branches,leaves,as,bs,zs)
    end


    function y_mat(y)
        ```
        Get y matrix as a matrix of categorical
        ```
        n = length(y)
        y_class = int(categorical(y),type=Int)
        Y = zeros(n,length(unique(y_class)))
        for i in 1:n, k in y_class
            if y_class[i] == k
                Y[i,k] = 1
            end
        end
        return(Y)
    end


    function subtree_inputs(Tt,x,y)
        #returns a list of indices of observations contained in leaf nodes of subtree
        Y = y_mat(y)
        obs = []
        for leaf in Tt.leaves
            append!(obs,[k for (k,v) in Tt.z if v==leaf])
        end
        return(obs)#,x[obs,:],Y[obs,:])
    end

end
