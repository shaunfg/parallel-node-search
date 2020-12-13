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

    using Random, DecisionTree, LinearAlgebra
    import MLJBase.int
    using CategoricalArrays


    function warm_start(depth::Int,y,x,seed; rf_ntrees = 1)
        n = size(x,1) #num observations
        p = size(x,2) #num features
        T = Tree([],[],[],Dict(),Dict(),Dict())
        e = 1*Matrix(I,p,p)
        #warmstart with randomForest
        seed = Random.seed!(seed)
        rf = build_forest(y,x,floor(Int,sqrt(p)),rf_ntrees,0.7,depth,5,2,rng = seed)
        obj = rf.trees[1]
        T = _branch_constraint(obj,1,T,e) #get the branching constraints from the randomForest
        T = _assign_class(x,T,e) #fill the z matrix with classses
        return T
    end

    function _branch_constraint(obj,node,t,e)
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

    function _assign_class(X,T,e)
        n = size(X,1)
        #for each observation, find leaf and assign class
        for i in 1:n
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

    N_nodes(D::Int) = 2^(D+1) - 1
    N_branch(D::Int) = Int(floor(N_nodes(D::Int)/2))
    get_parent(node) = Int(floor(node/2))

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

    function progenate(node,T)#::Int,T::Tree)
        #append leave nodes to tree struct
        branches = node
        leaves = [2*node,2*node+1]
        nodes = [node,2*node,2*node+1]
        return Tree(nodes,branches,leaves)
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
        nodes = _nodes_subtree(node,t)
        #return new struct containing subtree
        leaves = t.leaves[(in(nodes).(t.leaves))]
        branches = nodes[(.!(in(leaves).(nodes)))]
        bs = Dict(key => t.b[key] for key in branches)
        as = Dict(key => t.a[key] for key in branches)
        zs = Dict(k => t.z[k] for (k,v) in t.z if v in leaves)
        return Tree(nodes,branches,leaves,as,bs,zs)
    end


    function y_mat(y)
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

    function replace_subtree(t,subtree)#::Tree,subtree::Tree)
        st_nodes_f = subtree.nodes #get the root node of subtree
        st_root = minimum(st_nodes_f) #get the nodes of original subtree
        st_nodes_i = nodes_subtree(st_root,t) #delete nodes no longer optimal and add new nodes from optimal subtree
        keep_nodes = t.nodes[(.!(in(st_nodes_i).(t.nodes)))]
        append!(keep_nodes,st_nodes_f)
        #create new tree struct
        new_leaves = t.leaves[(.!(in(st_nodes_i).(t.leaves)))]
        new_branches = t.branches[(.!(in(st_nodes_i).(t.branches)))]
        for j in st_nodes_f
            if j in subtree.leaves
                append!(new_leaves,j)
            else
                append!(new_branches,j)
            end
        end
        return Tree(keep_nodes,new_branches,new_leaves)
    end


    function assign_class_subtree(X,T,a,b,e,z,indices)
        #for each observation, find leaf and assign class
        n = size(X,1)
        z = Dict()
        for i in indices
            #node = 1
            node = minimum(T.nodes)
            _cascade_down(node,X,i,T,z,a,b,e)
        end
        return(z)
    end


    function subtree_inputs(Tt::Tree,x,y)
        Y = y_mat(y)
        #returns a list of indices of observations contained in leaf nodes of subtree
        obs = []
        for leaf in Tt.leaves
            append!(obs,[k for (k,v) in Tt.z if v==leaf])
        end
        return(obs,x[obs,:],Y[obs,:])
    end

end
