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

    function get_tree(depth::Int)
        nodes = collect(1:N_nodes(depth))
        branches = collect(1:N_branch(depth))
        leaves = collect(N_branch(depth)+1:N_nodes(depth))
        return Tree(nodes,branches,leaves)
    end

    function replace_subtree(t::Tree,subtree::Tree)
        st_nodes_f = subtree.nodes
        #get the root node of subtree
        st_root = minimum(st_nodes_f)
        #get the nodes of original subtree
        st_nodes_i = nodes_subtree(st_root,t)
        #delete nodes no longer optimal and add new nodes from optimal subtree
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

end


Y = tf.tf.y_mat(["c1","c2","c1","c3"])
temp = tf.tf.get_tree(2)
temp.leaves
z = [1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]

function assign_class(leaf_node::Int, Y::Matrix, z::Matrix)
    n = size(Y)[1]
    K = size(Y)[2]

end
