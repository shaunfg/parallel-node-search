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

    function get_right_ancestors(node::Int)
        right_ancestors = []
        if node==1
            return()
        elseif (node-1)/2==get_parent(node)
            append!(right_ancestors,get_parent(node))
            append!(right_ancestors,get_right_ancestors(get_parent(node)))
        end
        return(right_ancestors)
    end

    function get_left_ancestors(node::Int)
        left_ancestors = []
        if node==1
            return()
        elseif node/2==get_parent(node)
            append!(left_ancestors,get_parent(node))
            append!(left_ancestors,get_left_ancestors(get_parent(node)))
        end
        return(left_ancestors)
    end



    function get_tree(depth::Int)
        nodes = collect(1:N_nodes(depth))
        branches = collect(1:N_branch(depth))
        leaves = collect(N_branch(depth)+1:N_nodes(depth))
        return Tree(nodes,branches,leaves)
    end
end

z = [1 2; 3 4; 3 4]
size(z)

function leaf_prediction(leaf_node,z,y)
    K = size(y)[2]
    n = size(y)[1]
    z_leaf = z[:,leaf_node]
    dominant_class = 1
    y'*z_leaf
end





function y_mat(y)
    n = length(y)
    y_class = int(categorical(y),type=Int)
    Y = zeros(n,k)
    for i in 1:n, k in y_class
        if y_class[i] == k
            Y[i,k] = 1
        end
    end
    return(Y)
end

y = ["c1","c2","c3","c1"]
Y = y_mat(y)
