using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
include("tree.jl")

iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

#-------------

T_output = LocalSearch(x,y,2;seed= 100)

function LocalSearch(x,y,tdepth;seed = 100,tol_limit = 1e-5)
    println("##############################")
    println("### Local Search Algorithm ###")
    println("##############################")

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
        local shuffled_t = shuffle(T.nodes)
        for t in shuffled_t

            println("STARTING TREE node $t-- $T")
            global Tt = tf.create_subtree(t,T)
            println("STARTING node $t-- $Tt")
            test_tree(Tt)
            local indices = tf.subtree_inputs(Tt,X,y)
            println(length(indices))
            global Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T,e)
            test_tree(Ttnew)
            if better_found ==true
                T = replace_subtree(T,Ttnew,X;print_prog=true)
                test_tree(T)
                global output = T
            end
            Lcur = loss(T,y)
            tol = abs(Lprev - Lcur)
            println("Lprev $Lprev, Lcur $Lcur")
        end
        println("Tolerance = $tol, Error = $Lcur, starting error = $starting_loss")
    end
    return T
end

function replace_subtree(T_full,subtree,X; print_prog = false)#::Tree,subtree::Tree
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
    else #
        parent = minimum(subtree.nodes)
        if print_prog == true
            println("replacement - branch $parent")
        end
        lower = parent *2
        upper = parent *2 + 1
        children = tf.get_children(parent,T)
        if isnothing(children)==false
            filter!(x->x ∉ children,T.nodes) # remove kid from nodes
            filter!(x->x ∉ children,T.branches) # remove kid from branches
            filter!(x->x ∉ children,T.leaves) # remove kid from leaves
            filter!(p -> p.first ∉ children, T.b)
            filter!(p -> p.first ∉ children, T.a)
        end
        if parent ∉ T.branches
            append!(T.branches,parent)
        end
        sub_children = tf.get_children(parent,subtree) # subtree children
        append!(T.nodes,sub_children)
        append!(T.leaves,sub_children)
        filter!(x -> x ≠ parent, T.leaves) # incase replacing on leaf
        T.a[parent] = subtree.a[parent]
        T.b[parent] = subtree.b[parent]
        for point in keys(subtree.z) # add assignments of z to parent
            T.z[point] = subtree.z[point]
        end
    end
    return(T)
end

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
    return (L) #so that it not so small for now
    # return (L/L̂ + α*C) #so that it not so small for now
    # return(L + α*C*L̂)
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
function optimize_node_parallel(Tt,indices,X,y,T,e)
    Y =tf.y_mat(y)
    XI = X[indices,:]
    YI = Y[indices,:]
    root = minimum(Tt.nodes)
    Tnew = Tt
    Tbest = Tt

    better_split_found = false

    error_best = loss(Tt,y)
    println("(Node $root)")
    if root in Tt.branches
        println("Optimize-Node-Parallel : Branch split")
        println("Tt     $Tt")
        Tnew = tf.create_subtree(root,Tt)
        Tnew = 
        println("New tree $Tnew")
        Tlower_sub = tf.create_subtree(tf.left_child(root,Tt),Tt)
        Tupper_sub = tf.create_subtree(tf.right_child(root,Tt),Tt)
    else
        println("Optimize-Node-Parallel : Leaf split")
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
        Tlower_sub = tf.create_subtree(tf.left_child(root,Tnew),Tnew)
        Tupper_sub = tf.create_subtree(tf.right_child(root,Tnew),Tnew)
    end
    Tlower = replace_subtree(Tnew,Tlower_sub,X)
    Tupper = replace_subtree(Tnew,Tupper_sub,X)
    Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y)

    println("Para tree $Tpara")
    error_lower = loss(Tlower,y)
    error_upper = loss(Tupper,y)

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

function new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
    branches = [root]
    leaves = [2*root,2*root+1]
    nodes = [root,2*root,2*root+1]
    n,p = size(XI)
    Tfeas = tf.Tree(nodes,branches,leaves,Dict(),Dict(),Dict())

    new_values, error_best = _get_split(root,XI,YI,Tfeas,e,indices,X,y;Nmin=5)
    Tpara = tf.copy(Tfeas)
    Tpara.a[root] = Int(new_values[1])
    Tpara.b[root] = new_values[2]
    filter!(x->x≠root, Tpara.leaves)

    Tpara = tf.assign_class(X,Tpara,e;indices = indices)
    return Tpara
end

function best_parallelsplit(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    #println("test- in parallel split")
    new_values, error_best = _get_split(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    Tpara = tf.copy(Tt)
    Tpara.a[root] = Int(new_values[1])
    Tpara.b[root] = new_values[2]
    filter!(x->x≠root, Tpara.leaves)
    return(Tpara,error_best)
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
        if tf.minleafsize(Tttry) >= Nmin && error < error_best && true == true
            error_best = error
            new_values = [j,bsplit]
            better_split_found = true
        end
    end
    if better_split_found == true
        println("FOUND FEASIBEL TREE")
        return(new_values,error_best)
    else
        println("NO FEASIBLE TREE")
        return([1,0.5],loss(Tt,y))
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

T = tf.warm_start(10,y,x,100)

Tt = tf.create_subtree(3,T)

test_tree(replace_subtree(T,Tt,X))
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
    if isempty(T.a) == false
        @test keys(T.a) ⊈ T.leaves
        @test keys(T.b) ⊈ T.leaves
    end
    @test isempty(T.z) == false
end

test_tree(T)

function test_replace_subtree(T_new,T_old)
    test_tree(T_new)
    test_tree(T_old)
end

test_replace_subtree()

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
