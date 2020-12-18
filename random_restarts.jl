ncores = length(Sys.cpu_info());
Threads.nthreads()
cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")
include("tree.jl")

function threaded!(x0,bifer_out_thread,ff,r_iter,num_attract;warmup=400)
    #we need to perform the calculations separately on remaining columns when there are remainder columns
    Threads.@threads for i in 1:Threads.nthreads()
        @inbounds begin
            indices = 1+(i-1)*cols_per_thread:cols_per_thread*i
            for j in indices
                bifer_view = @view bifer_out_thread[1:num_attract,j]
                calc_attractor!(x0,bifer_view,ff,r_iter[j],num_attract;warmup=400)
            end
        end
    end
end

@btime threaded!(x0,bifer_out_thread,ff,r,num_attract;warmup=400)



#parallelize random restarts
nrestarts = 20
function threaded_restarts!(x,y,nrestarts;warmup=400)
    numthreads = Threads.nthreads()-4
    restarts_per_thread = Int(nrestarts/numthreads)
    seed_values = 100:100:100*nrestarts
    #we need to perform the calculations separately on remaining columns when there are remainder columns
    Threads.@threads for i in 1:numthreads
        indices = 1+(i-1)*restarts_per_thread:restarts_per_thread*i
        for j in indices
            seed = seed_values[j]
            println(seed)
            LocalSearch(x,y,2,seed)
        end
    end
end
threaded_restarts!(x,y,nrestarts;warmup=400)



ParallelLocalSearch(x,y,2,100)

function ParallelLocalSearch(x,y,tdepth,seed;tol_limit = 1e-5)
    println("##############################")
    println("### Local Search Algorithm ###")
    println("##############################")

    function half_parallelsearch(halftree_nodes,T,X,y)
        local shuffled_t = shuffle(halftree_nodes)
        for t in shuffled_t
            # println("STARTING TREE node $t-- $T")
            # global Tt = tf.create_subtree(t,T)
            # println("STARTING node $t-- $Tt")
            # #test_tree(Tt)
            # local indices = tf.subtree_inputs(Tt,X,y)
            # println(length(indices))
            # global Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T,e)
            # #test_tree(Ttnew)
            # if better_found ==true
            #     T = replace_subtree(T,Ttnew,X;print_prog=true)
            #     #test_tree(T)
            #     global output = T
            # end
            # Lcur = loss(T,y)
            # tol = abs(Lprev - Lcur)
            # println("Lprev $Lprev, Lcur $Lcur")
            ## TODO - what to return, what to define global, how to replace both left and right subtrees at the same time
        end
    end


    # Data pre-processing - Normalisation
    dt = fit(UnitRangeTransform, x, dims=1)
    X = StatsBase.transform(dt,x)
    e = tf.get_e(size(x,2))
    local T= tf.warm_start(tdepth,y,X,seed)
    starting_loss = loss(T,y)
    tol = 10
    local iter = 0
    while tol > tol_limit
        iter +=1
        println("------------- Iteration # $iter ----------------")
        Lprev = loss(T,y)
        local Lcur

        #Break the trees into left and right half
            ## nodes, subtree, and indices in left half of tree
            left_half_nodes = tf._nodes_subtree(2,T)
            global Tt_L = tf.create_subtree(2,T)
            local L_indices = tf.subtree_inputs(Tt_L,X,y)
            ## nodes, subtree, and indices in right half of tree
            right_half_nodes = tf._nodes_subtree(3,T)
            global Tt_R = tf.create_subtree(3,T)
            local R_indices = tf.subtree_inputs(Tt_R,X,y)
            numthreads = Threads.nthreads()-4
        #we need to perform the calculations separately on remaining columns when there are remainder columns
        Threads.@threads for i in 1:numthreads
            if i == 1
                half_parallelsearch(left_half_nodes,Tt_L,X[L_indices,:],y[L_indices,:])
            else
                half_parallelsearch(right_half_nodes,Tt_R,X[R_indices,:],y[R_indices,:])
            end
        end
        tol = 1e-10
        #println("Tolerance = $tol, Error = $Lcur, starting error = $starting_loss")
    end
    return T
end




#Tune hyperparameters
function tune(dmax)
    vbest = Inf
    Cp_vals = [0.001, 0.01, 0.1, 1, 10]
    d_vals = collect(1:dmax)
    for d in d_vals, α in Cp_vals
        #get errors, Cp value, depth from tree
        #v = LocalSearch(x,y,d,α,seed)
        if v < vbest
            vbest = v
            αbest = α
            Dbest = d
        end
    end
    return(αbest,Dbest)
end
