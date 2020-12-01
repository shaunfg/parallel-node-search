"""
Functions and stucture needed to describe a tree, this is used to create
an object to supplement the optimal classification tree.
"""
module tf
    struct Tree
        nodes
        branches
        leaves
#        a #pxm matrix recording which variables being split on in each branch
#        b #m matrix recording split thresholds at each branch
    end

    using Random

    function get_randtree(depth::Int,p::Int)
        nodes = collect(1:N_nodes(depth))
        branches = collect(1:N_branch(depth))
        leaves = collect(N_branch(depth)+1:N_nodes(depth))
        # m = length(branches)
        # b = rand(m,1) #randomize split thresholds for each branch
        # a = zeros(p,m) #features being splitted in each branch
        # for j in 1:m
        #     i = rand(1:p)
        #     a[i,j] = 1
        # end
        return Tree(nodes,branches,leaves)
    end

    N_nodes(D::Int) = 2^(D+1) - 1
    N_branch(D::Int) = Int(floor(N_nodes(D::Int)/2))

    # get_left(node) = 2*node
    # get_right(node) =  2*node + 1
    get_parent(node) = Int(floor(node/2))

    function left_child(node::Int,T::Tree)
        if 2*node in T.nodes
            return(2*node)
        end
    end

    function right_child(node::Int,T::Tree)
        if 2*node+1 in T.nodes
            return(2*node+1)
        end
    end

    function progenate(node::Int,T::Tree)
        branches = vcat(T.branches,node)
        leaves = vcat(T.leaves,2*node,2*node+1)
        nodes = vcat(T.nodes,2*node,2*node+1)
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

    function nodes_subtree(node::Int,t::Tree)
        # Get Subtree
        subtree_nodes = []
        subtree_leaves = []
        if node in t.leaves
            append!(subtree_nodes,node)
            append!(subtree_leaves,node)
        else
            append!(subtree_nodes,node)
            append!(subtree_nodes,nodes_subtree(left_child(node,t),t))
            append!(subtree_nodes,nodes_subtree(right_child(node,t),t))
        end
        return(subtree_nodes)
    end

    function create_subtree(nodes,t::Tree)
        leaves = []
        branches = copy(nodes)
        for d in nodes
            if d in t.leaves
                append!(leaves,d)
                filter!(x->x≠d,branches)
            end
        end
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

    function replace_subtree(t::Tree,subtree::Tree)
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

    function assign_class(X,T,a,b,z,e)
        n = size(X,1)
        for i in 1:n
            node = 1
            cascade_down(node,X,i,T,z,a,b,e)
        end
    end

    function cascade_down(node::Int,X,i::Int,T,z,a,b,e)
        if node in T.leaves
            assign_leaf(i,node,T,z)
        else
            j = findall(x->x==1, a[:,node])[1]
            if (e[:,j]'*X[i,:] < b[node])
                cascade_down(left_child(node,T),X,i,T,z,a,b,e)
            else
                cascade_down(right_child(node,T),X,i,T,z,a,b,e)
            end
        end
    end

    function assign_leaf(i::Int,leafnode::Int,T,z)
        z[i,leafnode] = 1
    end

end
