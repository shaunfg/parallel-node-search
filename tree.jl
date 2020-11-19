module tree_functions
    struct Tree
        nodes
        branches
        leaves
    end

    N_nodes(D::Int) = 2^(D+1) - 1
    N_branch(D::Int) = Int(floor(N_nodes(D::Int)/2))

    get_left(node) = 2*node
    get_right(node) =  2*node + 1

    function get_tree(depth::Int)
        nodes = collect(1:N_nodes(depth))
        branches = collect(1:N_branch(depth))
        leaves = collect(N_branch(depth)+1:N_nodes(depth))
        return Tree(nodes,branches,leaves)
    end
end  # module tree_functions


#------
# tree = get_tree(1)
# tree.leaves
# get_left(1)
# get_right(1)
# get_tree(2)
