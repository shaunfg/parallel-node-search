"""
Functions and stucture needed to describe a tree, this is used to create
an object to supplement the optimal classification tree.
"""

module tf
    struct Tree
        nodes
        branches
        leaves
    end

    N_nodes(D::Int) = 2^(D+1) - 1
    N_branch(D::Int) = Int(floor(N_nodes(D::Int)/2))

    get_left(node) = 2*node
    get_right(node) =  2*node + 1
    get_parent(node) = Int(floor(node/2))

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
end


Y = tf.tf.y_mat(["c1","c2","c1","c3"])
temp = tf.tf.get_tree(2)
temp.leaves
z = [1 0 0 0; 0 0 0 1; 0 0 1 0; 0 1 0 0]

function assign_class(leaf_node::Int, Y::Matrix, z::Matrix)
    n = size(Y)[1]
    K = size(Y)[2]

end
