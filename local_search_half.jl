using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
# cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")
include("tree.jl")



iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])
#---------------
lend_full = CSV.read("../lending-club/lend_training_70.csv",DataFrame)
lend = lend_full[randperm(size(lend_full,1)),:][1:500,:]
x = Matrix(select(lend,Not(:loan_status)))
y = Vector(lend[:,:loan_status])
#----- Profiling
using Profile
@profile LocalSearch(x,y,2,400,α=0.001)
Juno.profiler()

#-------------
T_output = LocalSearch(x,y,1,400,α=0.0001)

trees = threaded_restarts!(x,y,nrestarts;warmup=400)
ncores = length(Sys.cpu_info());
Threads.nthreads()

#parallelize random restarts
nrestarts = 8
function threaded_restarts!(x,y,nrestarts;warmup=400)
    numthreads = Threads.nthreads()-4
    restarts_per_thread = Int(nrestarts/numthreads)
    seed_values = 100:100:100*nrestarts
    output_tree = Dict()
    #we need to perform the calculations separately on remaining columns when there are remainder columns
    Threads.@threads for i in 1:numthreads
        indices = 1+(i-1)*restarts_per_thread:restarts_per_thread*i
        for j in indices
            seed = seed_values[j]
            println(seed)
            Tree = LocalSearch(x,y,2,seed)
            output_tree[j] = Tree
        end
    end
    return(output_tree)
end


function LocalSearch(x,y,tdepth,seed;tol_limit = 1,α=0.01)
    println("##############################")
    println("### Local Search Algorithm ###")
    println("##############################")

    function HalfTreeSearch(T,X,y,tol_limit)
        tol = 10
        local iter = 0
        while tol > tol_limit
            iter +=1
            println("------------- Iteration # $iter ----------------")
            Lprev = loss(T,y)
            local Lcur
            local shuffled_t = shuffle(T.nodes)
            print(shuffled_t)
            for t in shuffled_t
                if t ∈ T.nodes
                    # println("STARTING TREE node $t-- $T")
                    Tt = tf.create_subtree(t,T)
                    # println("STARTING node $t-- $Tt")
                    local indices = tf.subtree_inputs(Tt,X,y)
                    # println(length(indices))
                    Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T,e)
                    if better_found ==true
                        global T_replacement = Ttnew
                        T = replace_subtree(T,Ttnew,X;print_prog=true)
                        global output = T
                        # println("replaced Tree $t-- $T")
                    end
                    Lcur = loss(T,y)
                    tol = abs(Lprev - Lcur)
                    println("Lprev $Lprev, Lcur $Lcur")
                end
            end
            println("Tolerance = $tol, Error = $Lcur, starting error = $starting_loss")
        end
        return T
    end

    # Data pre-processing - Normalisation
    dt = fit(UnitRangeTransform, x, dims=1)
    X = StatsBase.transform(dt,x)
    e = tf.get_e(size(x,2))
    local T = tf.warm_start(tdepth,y,X,seed)
<<<<<<< HEAD:local_search.jl
    global starting_tree = T
    starting_loss = loss(T,y,α)
    tol = 10
    local iter = 0
    while tol > tol_limit
        iter +=1
        println("------------- Iteration # $iter ----------------")
        Lprev = loss(T,y,α)
        local Lcur
        local shuffled_t = shuffle(T.nodes)
        # print(shuffled_t)
        for t in shuffled_t
            if t ∈ T.nodes
                # println("STARTING TREE node $t-- $T")
                Tt = tf.create_subtree(t,T)
                # println("STARTING node $t-- $Tt")
                local indices = tf.subtree_inputs(Tt,X,y)
                # println(length(indices))
                Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T,e;α=α)
                if better_found ==true
                    T = replace_subtree(T,Ttnew,X;print_prog=true)
                    # global output = T
                    # println("replaced Tree $t-- $T")
                end
                Lcur = loss(T,y,α)
                tol = abs(Lprev - Lcur)
                println("Lprev $Lprev, Lcur $Lcur")
            end
=======
    global previous_tree = T
    starting_loss = loss(T,y)

    #Break the trees into left and right half: get nodes, subtrees, indices
    global Tt_L = tf.create_subtree(2,T)
    local L_indices = tf.subtree_inputs(Tt_L,X,y)

    global Tt_R = tf.create_subtree(3,T)
    local R_indices = tf.subtree_inputs(Tt_R,X,y)
    numthreads = Threads.nthreads()-4

    Threads.@threads for i in 1:numthreads
        if i == 1
            Tt_L = HalfTreeSearch(Tt_L,X,y,tol_limit)
        else
            Tt_R = HalfTreeSearch(Tt_R,X,y,tol_limit)
>>>>>>> dev/halftree:local_search_half.jl
        end
    end
    tol = 1e-10
    T = replace_subtree(T,Tt_R,X;print_prog=true)
    T = replace_subtree(T,Tt_L,X;print_prog=true)
    println("FInal output tree $T")
    global final_tree = T
    Lcur = loss(T,y)
    println("Lprev $starting_loss, Lcur $Lcur")

    test_tree(T)
end

get_level(node,subtree_root) = Int(floor(log2(node/subtree_root)))
# get_level(50,12)
calculate_destination(parent_root,subtree_root,node) = node + (parent_root-subtree_root)*2^(get_level(node,subtree_root))
# calculate_destination(51,3,12)

function replace_lower_upper(T_full,subtree,X; print_prog = false)#::Tree,subtree::Tree
    local T = tf.copy(T_full)
    if length(subtree.nodes) == 1 #
        kid = minimum(subtree.nodes)
        parent = tf.get_parent(kid)
        if print_prog == true
            println("replacement - leaf $kid --> $parent")
        end
        children = tf.get_children(parent,T)
        delete!(T.a,parent) # removed parent from branch
        delete!(T.b,parent)
        filter!(p -> p.first ∉ children, T.b)
        filter!(p -> p.first ∉ children, T.a)
        filter!(x->x ≠parent,T.branches) # remove parent from branches
        filter!(x->x ∉children,T.nodes) # remove kid from nodes
        filter!(x->x ∉children,T.branches) # remove kid from branches
        filter!(x->x ∉children,T.leaves) # remove kid from leaves
        append!(T.leaves,parent) # add parent to leaves
        points = [k for (k,v) in T.z if v in children]
        for point in points # add assignments of z to parent
            T.z[point] = parent
        end
    else
        subtree_parent = minimum(subtree.nodes) #7
        tree_parent = tf.get_parent(subtree_parent)#minimum(T.nodes) #3
        children = tf.get_children(tree_parent,T) #3 onwards
        CD(node) = calculate_destination(tree_parent,subtree_parent,node)
        filtered_nodes = [children;tree_parent]
        # println("filtered",filtered_nodes)
        filter!(x->x ∉ filtered_nodes, T.nodes) # remove kid from nodes
        filter!(x->x ∉ filtered_nodes,T.branches) # remove kid from nodes
        filter!(x->x ∉ filtered_nodes,T.leaves) # remove kid from nodes

        new_nodes = [CD(node) for node in subtree.nodes]
        new_branches = [CD(node) for node in subtree.branches]
        new_leaves = [CD(node) for node in subtree.leaves]

        # println("get children",subtree.nodes)
        # println(calculate_destination(tree_parent,subtree_parent,subtree.nodes))
        append!(T.nodes, new_nodes)
        append!(T.branches, new_branches)
        append!(T.leaves, new_leaves)
        # println(T.nodes)
        # test_tree(T)
        extra_branches = [k for (k,v) in T.b if k ∈ children]
        filter!(p -> p.first ∉ extra_branches, T.a)
        filter!(p -> p.first ∉ extra_branches, T.b)

        for key in keys(subtree.a)
            T.a[CD(key)] = subtree.a[key]
            T.b[CD(key)] = subtree.b[key]
        end

        e = tf.get_e(size(X,2))
        T = tf.assign_class(X,T,e)
    end
    return T
end

function replace_subtree(T_full,subtree,X; print_prog = false)#::Tree,subtree::Tree
    local T = tf.copy(T_full)

    parent =  minimum(subtree.nodes)
    children = tf.get_children(parent,T)
    # println(minimum(subtree.nodes))
    if print_prog == true
        println("replacement - branch $parent")
    end

    if isnothing(children)==false # must have children on T tree
        filter!(x->x ∉children,T.nodes) # remove kid from nodes
        filter!(x->x ∉children,T.branches) # remove kid from branches
        filter!(x->x ∉children,T.leaves) # remove kid from leaves
        filter!(p -> p.first ∉ children, T.a)
        filter!(p -> p.first ∉ children, T.b)
    end
    filter!(x->x ≠parent,T.leaves) # remove parent from branches
    filter!(x->x ≠parent,T.branches) # remove parent from branches
    filter!(x->x ≠parent,T.nodes) # remove parent from branches

    append!(T.leaves,subtree.leaves) # add parent to leaves
    append!(T.branches,subtree.branches) # add parent to leaves
    append!(T.nodes,subtree.nodes) # add parent to leaves
    for node in keys(subtree.b)
        T.b[node] = subtree.b[node] # add parent to leaves
        T.a[node] = subtree.a[node] # add parent to leaves
    end
    for point in keys(subtree.z) # add assignments of z to parent
        T.z[point] = subtree.z[point]
    end
    return(T)
end

# Γs=rand(Truncated(Normal(0.01,0.01),0,100),n_test)

# ncores=
# nthreads=Threads.nthreads()-1
# nthreads
#Tune hyperparameters
function tune(dmax,x,y;nthreads = length(Sys.cpu_info()) -1 )
    vbest = Inf
    Cp_vals = [0.001, 0.01, 0.1, 1, 10]
    d_vals = collect(1:dmax)

    grid = zeros(length(Cp_vals) * length(d_vals),4)
    local j=1
    for v in Cp_vals, u in d_vals
            grid[j,1] = u
            grid[j,2] = v
            j+=1
    end
    vars = @view grid[1:end,:]
    seed =1000
    segment = Int(floor(size(vars,1)/nthreads))
    index = [[i*segment for i=0:nthreads-1] ; Int(floor(size(vars,1)))]
    Threads.@threads for t in 1:nthreads+1
    @inbounds begin
            for i=index[t-1]+1:index[t]
                T = LocalSearch(x,y,Int(vars[i,1]),seed;tol_limit = 1,α=vars[i,2])
                vars[i,3] = loss(T,y,vars[i,2])
                vars[i,4] = Threads.threadid()
            end
        end
    end
    best = vars[argmin(vars[:,3]),:]
    # return (best[1],best[2])
    return vars
end

# a = rand(3,3)
#
# var_check = tune(5,x,y,nthreads=15)
# [[i,j] for (i,j) in zip([0.001, 0.01,3],[1,3,3])]
#-----
function loss(T,y,α)
    Y_full = tf.y_mat(y)
    L̂ = _get_baseline(Y_full)
    z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
    Y = Y_full[z_keys,:] # Reorganize Y values to z order
    z_values = collect(values(T.z)) #
    Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ T.leaves]
    Nt = length.([[k for (k,v) in T.z if v ==t] for t ∈ T.leaves])
    Lt = [Nt[t] - maximum(Nkt[t]) for t=1:length(T.leaves)]
    L = sum(Lt)#/L̂ #### + α*Cp......
    C = length(T.branches)
    # println("L = $(L/L̂), C = $(α*C)")
    # return (L) #so that it not so small for now
    return (L/L̂*1 + α*C) #so that it not so small for now
end

# TODO Can reduce this as there are no more indices ... tecnically
function _get_baseline(Y)
    Nt = sum((Y),dims=1)
    error = size(Y,1) - maximum(Nt)
    return error
end


#--- Optimize Node Parallel
# Input: Subtree T to optimize, training data X,y
# Output: Subtree T with optimized parallel split at root
function optimize_node_parallel(Tt,indices,X,y,T,e;α)
    Y =tf.y_mat(y)
    XI = X[indices,:]
    YI = Y[indices,:]
    root = minimum(Tt.nodes)
    Tnew = Tt
    Tbest = Tt

    better_split_found = false

    error_best = loss(Tt,y,α)
    println("(Node $root)")
    if root in Tt.branches
        println("Optimize-Node-Parallel : Branch split")
        # println("Tt     $Tt")
        Tnew = tf.create_subtree(root,Tt)
        println("Timing Marker 5")
        # println("New tree $Tnew")
        Tlower_sub = tf.create_subtree(tf.left_child(root,Tt),Tt)
        Tupper_sub = tf.create_subtree(tf.right_child(root,Tt),Tt)
        println("Timing Marker 4")
    else
        println("Optimize-Node-Parallel : Leaf split")
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5,α=α)
        # println("Tnew",Tnew)
        if Tnew == false
            return (Tt,false)
        end
        Tlower_sub = tf.create_subtree(tf.left_child(root,Tnew),Tnew)
        Tupper_sub = tf.create_subtree(tf.right_child(root,Tnew),Tnew)
    end
    println("Timing Marker 1")
    Tlower = replace_lower_upper(Tnew,Tlower_sub,X)
    println("Timing Marker 2")
    Tupper = replace_lower_upper(Tnew,Tupper_sub,X)
    Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y,α=α)

    # println("Para tree $Tpara")
    error_lower = loss(Tlower,y,α)
    error_upper = loss(Tupper,y,α)
    println("Timing Marker 3")

    if error_para < error_best
        println("!! Better Split Found : subtree")
        println("-->Error : $error_best => $error_para")
        Tt,error_best = Tpara,error_para
        better_split_found = true
    end

    if error_lower < error_best
        println("!! Better Split Found : lower")
        println("-->Error : $error_best => $error_lower")
        Tt,error_best = Tlower,error_lower
        better_split_found = true
    end

    if error_upper < error_best
        println("!! Better Split Found : upper")
        println("-->Error : $error_best => $error_upper")
        Tt,error_best = Tupper,error_upper
        better_split_found = true

    end
    return(Tt,better_split_found)
end

function new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5,α)
    branches = [root]
    leaves = [2*root,2*root+1]
    nodes = [root,2*root,2*root+1]
    n,p = size(XI)
    Tfeas = tf.Tree(nodes,branches,leaves,Dict(),Dict(),Dict())

    new_values, error_best = _get_split(root,XI,YI,Tfeas,e,indices,X,y;Nmin=5,α=α)
    # println("newvalues",new_values)
    if new_values == false # no feasible split found
        return false
    else
        Tpara = tf.copy(Tfeas)
        Tpara.a[root] = Int(new_values[1])
        Tpara.b[root] = new_values[2]
        filter!(x->x≠root, Tpara.leaves)
        # println("TFEASIBLE $Tpara",indices)
        Tpara = tf.assign_class(X,Tpara,e;indices = indices)
    end
    return Tpara
end

function best_parallelsplit(root,XI,YI,Tt,e,indices,X,y;Nmin=5,α)
    #println("test- in parallel split")
    new_values, error_best = _get_split(root,XI,YI,Tt,e,indices,X,y;Nmin=5,α)
    if new_values == false # no feasible split found
        return (Tt, loss(Tt,y,α))
    else
        Tpara = tf.copy(Tt)
        Tpara.a[root] = Int(new_values[1])
        Tpara.b[root] = new_values[2]
        filter!(x->x≠root, Tpara.leaves)
        return (Tpara,error_best)
    end
end

function _get_split(root,XI,YI,Tt,e,indices,X,y;Nmin=5,α)
    n,p = size(XI)
    error_best = Inf
    Tttry = tf.copy(Tt)
    filter!(x->x≠root, Tttry.leaves)
    better_split_found = false
    # local new_values = [1,0.5]
    # local error_best = loss(Tt,y)
    local new_values, error_best
    for j in 1:p, i in 1:n-1
        values = sort(XI[:,j])
        bsplit = 0.5*(values[i] + values[i+1])
        Tttry.a[root] = j
        Tttry.b[root] = bsplit
        Tttry = tf.assign_class(X,Tttry,e;indices = indices)
        #create a tree with this new a and b
        error = loss(Tttry,y,α)
        # println("MIN LEAF SIZE",tf.minleafsize(Tttry))
        if tf.minleafsize(Tttry) >= Nmin && error < error_best && true == true
            # println("MIN LEAF SIZE",tf.minleafsize(Tttry))
            global testin = Tttry
            error_best = error
            new_values = [j,bsplit]
            # println("newvalues before",new_values,indices)
            better_split_found = true
        end
    end
    if better_split_found == true
        # println("newvalues after",new_values)
        println("FOUND FEASIBLE TREE")
        return(new_values,error_best)
    else
        println("NO FEASIBLE TREE")
        return(false,false)
    end
end

#---- Testing

using Test

@test 1 ==  1

function run_tests()
    dt = fit(UnitRangeTransform, x, dims=1)
    X = StatsBase.transform(dt,x)
    e = tf.get_e(size(X,2))
    T = tf.warm_start(10,y,X,100)
    T = tf.assign_class(X,T,e)
    test_tree(T)
    for node in T.nodes
        println("node $node")
        Tt = tf.create_subtree(node,T)
        test_tree(Tt)

    end
    # T = tf.warm_start(2,y,x,100)

end

run_tests()

dt = fit(UnitRangeTransform, x, dims=1)
X = StatsBase.transform(dt,x)
e = tf.get_e(size(X,2))
# T = tf.Tree(Any[1,2,3,6,7,12,13,14,15,28,29],
#                 Any[1,3,6,7,14],
#                 Any[2, 12, 13,15,28,29],
#                 Dict{Any,Any}(6=> 2,3 => 4,1 => 3,7=>1,14 =>1),
#                 Dict{Any,Any}(6 => 0.5,3 => 0.6458,1 => 0.245,7=>0.5,14 =>0.5),
#                 Dict())
# T = tf.assign_class(X,T,e)
#
# tf.create_subtree(29,T)

T = tf.warm_start(2,y,x,100)
Tt = tf.create_subtree(3,tf.warm_start(3,y,x,100))
output = replace_lower_upper(T,Tt,x)

test_tree(output)
indices = tf.subtree_inputs(Tt,x,y)
Ttnew,bs = optimize_node_parallel(Tt,indices,x,y,T,tf.get_e(size(x,2)))

test_tree(Ttnew)

function test_tree(T)
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

test_tree(T)

function test_reeplace_subtree(T_new,T_old)
    test_tree(T_new)
    test_tree(T_old)
end

test_reeplace_subtree()

root = 2
indices_test = tf.subtree_inputs(Tt_test,x,y)
XI_test = X[indices_test,:]
YI_test = tf.y_mat(y)[indices_test,:]
Tt_test = new_feasiblesplit(root,XI_test,YI_test,Tt_test,test_e,indices_test,X;Nmin=5)
tf.assign_class(x,Tt_test,test_e)

root = 3
Tt = tf.create_subtree(root,T)
indices = tf.subtree_inputs(Tt,X,y)
Y =tf.y_mat(y)
# println(X)
XI = X[indices,:]
YI = Y[indices,:]
# root = minimum(Tt.nodes)
e = tf.get_e(size(x,2))
Tnew = Tt
Tbest = Tt

error_best = loss(Tt,y)
println("---------$root-----------")
Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
Tlnew = tf.left_child(root,Tnew)
Tunew = tf.right_child(root,Tnew)
# println("bs, expect $root,",Tnew.b)
Tlnewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tlnew])
Tunewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tunew])
Tlower = tf.Tree([Tlnew],[],[Tlnew],Tnew.a,Tnew.b,Tlnewz)
Tupper = tf.Tree([Tunew],[],[Tunew],Tnew.a,Tnew.b,Tunewz)
splitss = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y)

# Tt_out = new_feasiblesplit(4,XI_test,YI_test,Tt_test,test_e,indices_test,X;Nmin=5)


test = true
if test != false
    print("hi")
end


T= tf.warm_start(2,y,X,seed)
T= tf.warm_start(2,y,x,100)
loss(T,y)
# Tt = tf.create_subtree(3,T)
Tt = tf.create_subtree(2,T)
indices = tf.subtree_inputs(Tt,X,y)
Ttnew = optimize_node_parallel(Tt,indices,X,y,T)
#
# T_old = tf.Tree(Any[1, 2,3,6,7,12,13,14,15,28,29],
#                 Any[1,3,6,7,14],
#                 Any[2, 12, 13,15,28,29],
#                 Dict{Any,Any}(6=> 2,3 => 4,1 => 3,7=>1,14 =>1),
#                 Dict{Any,Any}(6 => 0.5,3 => 0.6458333333333333,1 => 0.24576271186440676,7=>0.5,14 =>0.5),
#                 Dict(10 => 2,85 => 13, 20 => 12,1=>15,2=>28,40=>29))


output = replace_subtree(T,Ttnew,X)
loss(output,y)

unique(values(output.z))
output.leaves
output.a

(loss(T,y),length(T.z))
# (loss(Tt,y),length(Tt.z))
(loss(Ttnew,y),length(Tt.z))
T_combined = replace_subtree(T,Ttnew,X)
(loss(T_combined,y),length(T_combined.z))


Thelp = tf.copy(T)
append!(Thelp.branches ,[1000000000])


Tlower_sub
Tnew.branches
filter!(x->x ≠2,Tnew.branches) # remove parent from branches
