using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase

include("tree.jl")
include("local_search.jl")

#-------------
# T_output = LocalSearch(x,y,2,400)

loss(T,α) = loss(T,Y,α)

function LocalSearch_half(x,y,tdepth,seed;α=0.01,tol_limit = 1,numthreads=4)
    global Y =tf.y_mat(y)

    function HalfTreeSearch(T,X,y,tol_limit)
        tol = 10
        local iter = 0
        @inbounds while tol > tol_limit
            iter +=1
            # println("------------- Iteration # $iter ----------------")
            Lprev = loss(T,α)
            local Lcur
            local shuffled_t = shuffle(T.nodes)
            # print(shuffled_t)
            @inbounds for t in shuffled_t
                if t ∈ T.nodes
                    # println("STARTING TREE node $t-- $T")
                    Tt = tf.create_subtree(t,T)
                    # println("STARTING node $t-- $Tt")
                    local indices = tf.subtree_inputs(Tt,X,y)
                    # println(length(indices))
                    Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T,e,α=α)
                    if better_found ==true
                        global T_replacement = Ttnew
                        T = replace_subtree(T,Ttnew,X;print_prog=true)
                        global output = T
                        # println("replaced Tree $t-- $T")
                    end
                    Lcur = loss(T,α)
                    tol = abs(Lprev - Lcur)
                    # println("Lprev $Lprev, Lcur $Lcur")
                end
            end
            println("$iter) Tolerance = $tol, Error = $Lcur, starting error = $starting_loss")
        end
        return T
    end

    # Data pre-processing - Normalisation
    dt = fit(UnitRangeTransform, x, dims=1)
    X = StatsBase.transform(dt,x)
    e = tf.get_e(size(x,2))
    local T = tf.warm_start(tdepth,y,X,seed)
    global previous_tree = T
    starting_loss = loss(T,α)

    #Break the trees into left and right half: get nodes, subtrees, indices
    global Tt_L = tf.create_subtree(2,T)
    local L_indices = tf.subtree_inputs(Tt_L,X,y)

    global Tt_R = tf.create_subtree(3,T)
    local R_indices = tf.subtree_inputs(Tt_R,X,y)
    # numthreads = Threads.nthreads()-4
    Threads.@threads for i in 1:numthreads
        if i == 1
            Tt_L = HalfTreeSearch(Tt_L,X,y,tol_limit)
        else
            Tt_R = HalfTreeSearch(Tt_R,X,y,tol_limit)
        end
    end
    tol = 1e-10
    T = replace_subtree(T,Tt_R,X;print_prog=true)
    T = replace_subtree(T,Tt_L,X;print_prog=true)
    # println("FInal output tree $T")
    global final_tree = T
    Lcur = loss(T,α)
    println("Lprev $starting_loss, Lcur $Lcur")
    return T

end
# @inline function loss(T::tf.Tree,Y_full::Array{Float64,2},α::Float64)
#     L̂ = _get_baseline(Y_full)
#     z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
#     Y = Y_full[z_keys,:] # Reorganize Y values to z order
#     z_values = collect(values(T.z)) #
#     Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ T.leaves]
#     Nt = length.([[k for (k,v) in T.z if v ==t] for t ∈ T.leaves])
#     Lt = [Nt[t] - maximum(Nkt[t]) for t=1:length(T.leaves)]
#     L = sum(Lt)#/L̂ #### + α*Cp......
#     C = length(T.branches)
#     # println("L̂ = $L̂,L = $L, C = $(α*C)")
#     # return (L) #so that it not so small for now
#     return (L/L̂*1 + α*C) #so that it not so small for now
# end
#
# LocalSearch_half(x,y,2,400,α=0.01,numthreads=4)
