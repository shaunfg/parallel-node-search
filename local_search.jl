using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
# cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")

include("tree.jl")

iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

#-------------
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
trees = threaded_restarts!(x,y,nrestarts;warmup=400)


T_output = LocalSearch(x,y,2,100)

function LocalSearch(x,y,tdepth,seed;tol_limit = 1)
    println("##############################")
    println("### Local Search Algorithm ###")
    println("##############################")

    # Data pre-processing - Normalisation
    dt = fit(UnitRangeTransform, x, dims=1)
    X = StatsBase.transform(dt,x)
    e = tf.get_e(size(x,2))
    local T = tf.warm_start(tdepth,y,X,seed)
    # global initial_Tree = T
    starting_loss = loss(T,y)
    tol = 10
    local iter = 0
    while tol > tol_limit
        iter +=1
        println("------------- Iteration # $iter ----------------")
        Lprev = loss(T,y)
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
                Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T,e)
                if better_found ==true
                    T = replace_subtree(T,Ttnew,X;print_prog=true)
                    # global output = T
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

function calculate_destination(tree_root,subtree_root, current_node)
    output_binary = digits(current_node,base=2)
    # println(output_binary)
    levelz = length(digits(subtree_root,base = 2)) - length(digits(tree_root,base = 2))
    # println(length(output_binary),levelz)
    # println("levels",length(output_binary)-(levelz-1))
    for i in length(output_binary):-1:length(output_binary)-(levelz-1)
        output_binary[i] -= 1
    end
    # print(check)
    destination = sum(output_binary.*2 .^(0:(length(output_binary)-1)))
    return destination
end


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
        if print_prog == true
            println("replacement - subtree")
        end
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

        # # println("get children",subtree.nodes)
        # # println(calculate_destination(tree_parent,subtree_parent,subtree.nodes))
        append!(T.nodes, new_nodes)
        append!(T.branches, new_branches)
        append!(T.leaves, new_leaves)
        # # println(T.nodes)
        # # test_tree(T)
        # println(T)
        for key in keys(subtree.a)
            T.a[CD(key)] = subtree.a[key]
            T.b[CD(key)] = subtree.b[key]
        end
        filter!(p -> p.first ∉ T.leaves, T.a)
        filter!(p -> p.first ∉ T.leaves, T.b)
        println(T)
        # global T_check = T
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
    println("filtered!")
    # if isnothing(children)== false
    #     filter!(p -> p.first ∉ children, T.b)
    #     filter!(p -> p.first ∉ children, T.a)
    # end
    filter!(x->x ≠parent,T.leaves) # remove parent from branches
    filter!(x->x ≠parent,T.branches) # remove parent from branches
    filter!(x->x ≠parent,T.nodes) # remove parent from branches
    # println("children $children")


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
        # lower = parent *2
        # upper = parent *2 + 1
        # if isnothing(children)==false
        #     filter!(x->x ∉ children,T.nodes) # remove kid from nodes
        #     filter!(x->x ∉ children,T.branches) # remove kid from branches
        #     filter!(x->x ∉ children,T.leaves) # remove kid from leaves
        #     filter!(p -> p.first ∉ children, T.b)
        #     filter!(p -> p.first ∉ children, T.a)
        # end
        #

        # if parent ∉ T.branches
        #     append!(T.branches,parent)
        # end
        # sub_children = tf.get_children(parent,subtree) # subtree children
        # append!(T.nodes,sub_children)
        # append!(T.leaves,sub_children)
        # filter!(x -> x ≠ parent, T.leaves) # incase replacing on leaf
        # T.a[parent] = subtree.a[parent]
        # T.b[parent] = subtree.b[parent]
end

# dicttest = Dict(2=>"A",5=>"B")


#-----
function loss(T,y;α=0.01)
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
    # return(L + α*C*L̂)
end

# TODO Can reduce this as there are no more indices ... tecnically
function _get_baseline(Y)
    Nt = sum((Y),dims=1)
    error = size(Y,1) - maximum(Nt)
    return error
end

#
# for key in keys(subtrees)
#     println("key",key)
#     test_tree(subtrees[key])
#     ttt = replace_lower_upper(Tnew,subtrees[key],X)
#     test_tree(ttt)
# end
#--- Optimize Node Parallel
# Input: Subtree T to optimize, training data X,y
# Output: Subtree T with optimized parallel split at root
function optimize_node_parallel(Tt,indices,X,y,T,e)
    Y =tf.y_mat(y)
    XI = X[indices,:]
    YI = Y[indices,:]
    root = minimum(Tt.nodes)
    # global Tnew = Tt
    Tbest = Tt

    better_split_found = false

    error_best = loss(Tt,y)
    println("(Node $root)")
    if root in Tt.branches
        println("Optimize-Node-Parallel : Branch split")
        # println("Tt     $Tt")
        Tnew = tf.create_subtree(root,Tt)
        # println("New tree $Tnew")
    else
        println("Optimize-Node-Parallel : Leaf split")
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
        # println("Tnew",Tnew)
        if Tnew == false
            return (Tt,false)
        end

        subtrees = Dict()
        subtrees[tf.left_child(root,Tnew)] = tf.create_subtree(tf.left_child(root,Tnew),Tnew)
        subtrees[tf.left_child(root,Tnew)] = tf.create_subtree(tf.right_child(root,Tnew),Tnew)
    end
    # Tlower = replace_lower_upper(Tnew,Tlower_sub,X)
    # Tupper = replace_lower_upper(Tnew,Tupper_sub,X)
    Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y)

    # # println("Para tree $Tpara")
    # error_lower = loss(Tlower,y)
    # error_upper = loss(Tupper,y)

    if error_para < error_best
        println("!! Better Split Found : subtree")
        println("-->Error : $error_best => $error_para")
        Tt,error_best = Tpara,error_para
        better_split_found = true
    end

    # Thread: Find optimal deeper subtree
    subtree_nodes = tf.get_children(root,Tnew)
    total_subtrees = length(subtree_nodes)
    numthreads = Threads.nthreads()-4
    subtrees_per_thread = Int(floor(total_subtrees/numthreads))
    nremaining_subtrees = total_subtrees-subtrees_per_thread*numthreads

    global subtrees = Dict()
    Tdeep = Dict()
    global error_deep = Dict()

    println("subtrees per thread: ",subtrees_per_thread)
    println("nremaining_subtrees: ",nremaining_subtrees)


    Threads.@threads for i in 1:numthreads
        #get all possible children subtrees
        #println("Threads: ", i, ": ",subtree_nodes)
        for st_nodes in subtree_nodes[1+subtrees_per_thread*(i-1):i*subtrees_per_thread]
            subtrees[st_nodes] = tf.create_subtree(st_nodes,Tnew)
            Tdeep[st_nodes] = replace_lower_upper(Tnew,subtrees[st_nodes],X)
            error_deep[st_nodes] = loss(Tdeep[st_nodes],y)
            if error_deep[st_nodes] < error_best
                println("!! Better Split Found : lower")
                println("-->Error : $error_best => $error_deep")
                Tt,error_best = Tdeep[st_nodes],error_deep[st_nodes]
                better_split_found = true
            end
        end

        if i <= nremaining_subtrees
            subtrees[st_nodes] = tf.create_subtree(subtree_nodes[subtrees_per_thread*numthreads+i],Tnew)
        end
    end

    return(Tt,better_split_found)
end

function new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
    branches = [root]
    leaves = [2*root,2*root+1]
    nodes = [root,2*root,2*root+1]
    n,p = size(XI)
    Tfeas = tf.Tree(nodes,branches,leaves,Dict(),Dict(),Dict())

    new_values, error_best = _get_split(root,XI,YI,Tfeas,e,indices,X,y;Nmin=5)
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

function best_parallelsplit(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    #println("test- in parallel split")
    new_values, error_best = _get_split(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    if new_values == false # no feasible split found
        return (Tt, loss(Tt,y))
    else
        Tpara = tf.copy(Tt)
        Tpara.a[root] = Int(new_values[1])
        Tpara.b[root] = new_values[2]
        filter!(x->x≠root, Tpara.leaves)
        return (Tpara,error_best)
    end
end

function _get_split(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
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
        error = loss(Tttry,y)
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
