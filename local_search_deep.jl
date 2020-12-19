# include("tree.jl")
# include("local_search.jl")


function serial_deep!(x::Array{Float64,2},y,nrestarts::Int64,tdepth::Int64;α=0.001,tol_limit = 1e-4,warmup=400)
    seed_values = 100:100:100*nrestarts
    output_tree = Dict()
    #we need to perform the calculations separately on remaining columns when there are remainder columns
    indices = 1:nrestarts
    for j in indices
        seed = seed_values[j]
        #println(seed)
        Tree = LocalSearch(x,y,tdepth,seed,α=α,deep=true,tol_limit=tol_limit)
        output_tree[j] = Tree
    end
    return(output_tree)
end

# LocalSearch(x,y,3,400,α=0.01,deep=true,tol_limit=0.1)

function optimize_node_parallel_deep(Tt::tf.Tree,indices::Array{Int64,1},
            X::Array{Float64,2},y::Array{String,1},T::tf.Tree,e::Array{Int64,2},
            α::Float64;n_threads=10)
    Y =tf.y_mat(y)
    XI = X[indices,:]
    YI = Y[indices,:]
    root = minimum(Tt.nodes)
    Tbest = Tt

    better_split_found = false

    error_best = loss(Tt,α)
    # println("(Node $root)")
    local subtree_nodes,subtrees
    if root in Tt.branches
        global Tnew = tf.create_subtree(root,Tt)
        subtree_nodes = tf.get_children(root,Tnew)

    else
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5,α=α)
        if Tnew == false
            return (Tt,false)
        end
        subtree_nodes = tf.get_children(root,Tnew)
    end

    Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y,α=α)

    if error_para < error_best
        # println("!! Better Split Found : subtree")
        # println("-->Error : $error_best => $error_para")
        Tt,error_best = Tpara,error_para
        better_split_found = true
    end

    total_subtrees = length(subtree_nodes)
    errors = zeros(length(subtree_nodes),2)
    err = @view errors[1:end,:]
    nodes = @view subtree_nodes[1:end,:]
    threads_idx = tf.get_thread_idx(n_threads,subtree_nodes)
    @inbounds Threads.@threads for t in 1:n_threads
        if t <= length(subtree_nodes)
            for i=threads_idx[t]+1:threads_idx[t+1]
                err[i,1] = i
                err[i,2] = loss(replace_lower_upper(Tnew, tf.create_subtree(nodes[i],Tnew),X),α)
            end
        end
    end
    # println("AFTER THREADING",subtree_nodes)

    error_new = minimum(errors[:,2])
    new_key = [errors[i,1] for i=1:size(errors,1) if errors[i,2] == error_new][1]
    if error_new < error_best
        new_tree = replace_lower_upper(Tnew, tf.create_subtree(nodes[Int(new_key)],Tnew),X)
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
