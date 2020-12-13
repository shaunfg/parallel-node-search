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
        e = get_e(p)
        #warmstart with randomForest
        seed = Random.seed!(seed)
        rf = build_forest(y,x,floor(Int,sqrt(p)),rf_ntrees,0.7,depth,5,2,rng = seed)
        obj = rf.trees[1]
        T = _branch_constraint(obj,1,T,e) #get the branching constraints from the randomForest
        T = assign_class(x,T,e) #fill the z matrix with classses
        return T
    end

    get_e(p) = 1*Matrix(I,p,p)

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

    function assign_class(X,T,e;indices = false)
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

    function progenate(node,Tt,e,XI,YI,indices,X;Nmin=5) #::Int,T::Tree)
        #append leave nodes to tree struct
        new_tree = new_feasiblesplit(node,XI,YI,Tt,e,indices,X;Nmin=Nmin)
        # println("----newnodes",new_tree.nodes)
        return new_tree
    end

    # function new_feasiblesplit(root,XI,YI,Tt,anew,bnew,e,z,indices,X;Nmin=5)
    function new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
        # println("test- in parallel split")
        branches = root
        leaves = [2*root,2*root+1]
        nodes = [root,2*root,2*root+1]
        n,p = size(XI)
        Tfeas = Tree(branches,leaves,nodes,Dict(),Dict(),Dict())
        # println("----newnodes2  ",nodes)
        Infeasible = true
        while Infeasible
            for j in 1:p
                values = sort(XI[:,j])
                for i in 1:n-1
                    bsplit = 0.5*(values[i] + values[i+1])
                    Tfeas = tf.assign_class(X,Tfeas,e;indices = indices)
                    Tfeas.a[root] = j
                    Tfeas.b[root] = bsplit
                    minsize = minleafsize(Tt)
                    if (minsize >= Nmin)
                        Infeasible = false
                        return Tfeas
                    end
                end
            end
        end
    end

    function minleafsize(T)
        minbucket = Inf
        Nt = zeros(maximum(T.nodes))
        z_mat = zeros(maximum([k for (k,v) in T.z]),maximum(T.nodes))
        for t in T.leaves
            i = [k for (k,v) in T.z if v==t]
            z_mat[i,t] .= 1
            Nt[t] = sum(z_mat[:,t])
            if (Nt[t] > 0) & (Nt[t] < minbucket)
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

    function _nodes_subtree(node,t)#::Int,t::Tree)
        # Get Subtree of a specified node
        subtree_nodes = []
        subtree_leaves = []
        if node in t.leaves
            append!(subtree_nodes,node)
            append!(subtree_leaves,node)
        else
            println("marker34")
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
        # println(subtree)
        println("t= ",st_root)
        st_nodes_i = _nodes_subtree(st_root,t) #delete nodes no longer optimal and add new nodes from optimal subtree
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
        # println(t.a)
        # append the right a,b,z values.
        return Tree(keep_nodes,new_branches,new_leaves,t.a,t.b,t.z)
    end




    function subtree_inputs(Tt::Tree,x,y)
        Y = y_mat(y)
        #returns a list of indices of observations contained in leaf nodes of subtree
        obs = []
        for leaf in Tt.leaves
            append!(obs,[k for (k,v) in Tt.z if v==leaf])
        end
        return(obs)#,x[obs,:],Y[obs,:])
    end

end