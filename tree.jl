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

    function get_tree(depth::Int)
        nodes = collect(1:N_nodes(depth))
        branches = collect(1:N_branch(depth))
        leaves = collect(N_branch(depth)+1:N_nodes(depth))
        return Tree(nodes,branches,leaves)
    end
end  
