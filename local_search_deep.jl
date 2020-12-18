include("tree.jl")
# include("local_search.jl")
function optimize_node_parallel_deep(Tt::tf.Tree,indices::Array{Int64,1},
            X::Array{Float64,2},y::Array{String,1},T::tf.Tree,e::Array{Int64,2},
            α::Float64;n_threads=4)
    Y =tf.y_mat(y)
    XI = X[indices,:]
    YI = Y[indices,:]
    root = minimum(Tt.nodes)
    # global Tnew = Tt
    Tbest = Tt

    better_split_found = false

    error_best = loss(Tt,α)
    println("(Node $root)")
    local subtree_nodes
    if root in Tt.branches
        # println("Optimize-Node-Parallel : Branch split")
        # println("Tt     $Tt")
        global Tnew = tf.create_subtree(root,Tt)
        # println("New tree $Tnew")
        global subtrees = Dict()
        subtree_nodes = tf.get_children(root,Tnew)

    else
        # println("Optimize-Node-Parallel : Leaf split")
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5,α=α)
        # println("Tnew",Tnew)
        if Tnew == false
            return (Tt,false)
        end

        subtrees = Dict()
        subtrees[tf.left_child(root,Tnew)] = tf.create_subtree(tf.left_child(root,Tnew),Tnew)
        subtrees[tf.left_child(root,Tnew)] = tf.create_subtree(tf.right_child(root,Tnew),Tnew)
        subtree_nodes = tf.get_children(root,Tnew)
    end

    Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y,α=α)

    if error_para < error_best
        # println("!! Better Split Found : subtree")
        println("-->Error : $error_best => $error_para")
        Tt,error_best = Tpara,error_para
        better_split_found = true
    end

    total_subtrees = length(subtree_nodes)
    # numthreads = Threads.nthreads()-4
    # subtrees_per_thread = Int(floor(total_subtrees/numthreads))
    # nremaining_subtrees = total_subtrees-subtrees_per_thread*numthreads
    Tdeep = Dict()
    errors = zeros(length(subtree_nodes),2)
    err = @view errors[1:end,:]
    # global Tbest = Dict()
    # global error_best = Dict()
    # println("subtree_nodes",subtree_nodes)
    n_threads = 4#Threads.nthreads()-4
    threads_idx = tf.get_thread_idx(n_threads,subtree_nodes)
    # println(threads_idx,subtree_nodes)
    @inbounds Threads.@threads for t in 1:n_threads
        if t <= length(subtree_nodes)
            for i=threads_idx[t]+1:threads_idx[t+1]
                st_node = subtree_nodes[i]
                # println("count",st_node)
                subtrees[st_node] = tf.create_subtree(st_node,Tnew)
                # test_tree(subtrees[st_node])
                Tdeep[st_node] = replace_lower_upper(Tnew,subtrees[st_node],X)
                # test_tree(Tdeep[st_node])
                err[i,1] = st_node
                err[i,2] = loss(Tdeep[st_node],α)
            end
        end
    end

    error_new = minimum(errors[:,2])
    new_key = [errors[i,1] for i=1:size(errors,1) if errors[i,2] == error_new][1]
    new_tree = Tdeep[Int(new_key)]
    if error_new < error_best
        # println(Tbest)
        Tt,error_best = new_tree,error_new
        better_split_found = true
    end

    return(Tt,better_split_found)
end
using Test

function test_tree(T::tf.Tree)
    @test T.leaves ⊆ T.nodes
    @test T.branches ⊆ T.nodes
    @test vcat(T.leaves,T.branches) ⊆ T.nodes
    @test T.nodes ⊆ vcat(T.leaves,T.branches)
    @test unique(values(T.z)) ⊆ T.leaves
    @test T.leaves ⊆ unique(values(T.z))
    @test unique(values(T.z)) ⊈ T.branches
    @test keys(T.b) ⊆ T.branches
    @test keys(T.a) ⊆ T.branches
    @test length(T.nodes) == length(unique(T.nodes))
    @test length(T.leaves) == length(unique(T.leaves))
    @test length(T.branches) == length(unique(T.branches))
    if isempty(T.a) == false
        @test keys(T.a) ⊈ T.leaves
        @test keys(T.b) ⊈ T.leaves
    end
    @test isempty(T.z) == false
end
