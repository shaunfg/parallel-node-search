"""
Functions and stucture needed to describe a tree, this is used to create
an object to supplement the optimal classification tree.
"""
module tf
    struct Tree
        nodes
        branches
        leaves
        #a #pxm matrix recording which variables being split on in each branch
        #b #m matrix recording split thresholds at each branch
    end

    using Random, DecisionTree, LinearAlgebra

    function warm_start(depth::Int,y,x,seed)
    # function warm_start(depth::Int,p::Int,z,e,y,x)
        nodes = [] #collect(1:N_nodes(depth))
        branches = []
        leaves = []
        #branches = collect(1:N_branch(depth))
        #leaves = collect(N_branch(depth)+1:N_nodes(depth))
        T = Tree(nodes,branches,leaves)

        n = size(x,1) #num observations
        p = size(x,2) #num features
        e = 1*Matrix(I,p,p) #Identity matrix
        # a = zeros(p,m_nodes)
        # b = zeros(m_nodes)
        #z = zeros(n,length(T.nodes))
        a = Dict()
        b = Dict()
        z = Dict()


        #warmstart with randomForest
        rf_ntrees = 1
        seed = Random.seed!(seed)
        rf = build_forest(y,x,floor(Int,sqrt(p)),rf_ntrees,0.7,depth,5,2,rng = seed)

        function branch_constraint(obj,node,a,b,z)
            append!(nodes,node)
            if typeof(obj) != Leaf{String} #if the node is a branch get the branching constraints
                #a[obj.featid,node] = 1
                #b[node] = obj.featval
                a[node] = obj.featid
                b[node] = obj.featval
                append!(branches,node)
                branch_constraint(obj.left,2*node,a,b,z)
                branch_constraint(obj.right,2*node+1,a,b,z)
            else
                append!(leaves,node)
            end
        end

        obj = rf.trees[1]
        branch_constraint(obj,1,a,b,z) #get the branching constraints from the randomForest
        z = assign_class(x,T,a,b,e,z) #fill the z matrix with classses
        return (Tree(nodes,branches,leaves),a,b,z,e)
    end

    N_nodes(D::Int) = 2^(D+1) - 1
    N_branch(D::Int) = Int(floor(N_nodes(D::Int)/2))

    # get_left(node) = 2*node
    # get_right(node) =  2*node + 1
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

    function nodes_subtree(node,t)#::Int,t::Tree)
        # Get Subtree of a specified node
        subtree_nodes = []
        subtree_leaves = []
        if node in t.leaves
            append!(subtree_nodes,node)
            append!(subtree_leaves,node)
        else
            append!(subtree_nodes,node)
            append!(subtree_nodes,nodes_subtree(tf.left_child(node,t),t))
            append!(subtree_nodes,nodes_subtree(tf.right_child(node,t),t))
        end
        return(subtree_nodes)
    end

    function create_subtree(nodes,t)#,t::Tree)
        #return new struct containing subtree
        leaves = t.leaves[(in(nodes).(t.leaves))]
        branches = nodes[(.!(in(leaves).(nodes)))]
        # for d in nodes
        #     if d in t.leaves
        #         append!(leaves,d)
        #         t.leaves[(.!(in(st_nodes_i).(t.leaves)))]
        #         filter!(x->x≠d,branches)
        #     end
        # end
        return Tree(nodes,branches,leaves)
    end


    import MLJBase.int
    using CategoricalArrays

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

    function assign_class(X,T,a,b,e,z)
        n = size(X,1)
        z = Dict()
        #for each observation, find leaf and assign class
        for i in 1:n
            #node = 1
            node = minimum(T.nodes)
            cascade_down(node,X,i,T,z,a,b,e)
        end
        return(z)
    end

    function assign_class_subtree(X,T,a,b,e,z,indices)
        #for each observation, find leaf and assign class
        n = size(X,1)
        z = Dict()
        for i in indices
            #node = 1
            node = minimum(T.nodes)
            #println("Starting node :",node)
            cascade_down(node,X,i,T,z,a,b,e)
        end
        return(z)
    end

    function cascade_down(node::Int,X,i::Int,T,z,a,b,e)
        #cascade observation down the split
        if node in T.leaves
            #println("Node: ",node)
            assign_leaf(i,node,T,z)
        else
            j = a[node]
            if (e[:,j[1]]'*X[i,:] < b[node])
                #println("X3 value: ",X[i,j])
                tf.cascade_down(tf.left_child(node,T),X,i,T,z,a,b,e)
            else
                tf.cascade_down(tf.right_child(node,T),X,i,T,z,a,b,e)
            end
        end
    end

    function assign_leaf(i::Int,leafnode::Int,T,z)
        #assign observation to leaf
        #z[i,leafnode] = 1
        z[i] = leafnode
    end

    function subtree_inputs(Tt::Tree,z,x,y)
        #returns a list of indices of observations contained in leaf nodes of subtree
        obs = []
        for leave in Tt.leaves
            append!(obs,[k for (k,v) in z if v==leave])
            #append!(obs,findall(i->i==1, z[:,leave]))
        end
        # XI = x[indices,:]
        # yi = y[indices]
        return(obs,x[obs,:],y[obs,:])
    end

end
