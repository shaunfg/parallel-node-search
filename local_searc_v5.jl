using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
include("tree_ls_v2.jl")

iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

# Data pre-processing - Normalisation


#-------------
tdepth = 1
seed = 100

dt = fit(UnitRangeTransform, x, dims=1)
X = StatsBase.transform(dt,x)


println("##############################")
println("### Local Search Algorithm ###")
println("##############################")
T= tf.warm_start(tdepth,y,X,seed)
starting_loss = loss(T,y)
tol = 10
iter = 0
while tol >1e-5
    global iter +=1
# for i =1:1
    println("------------- Iteration # $iter ----------------")
    # println("current $i")
    Lprev = loss(T,y)
    local Lcur
    local shuffled_t = shuffle(T.nodes)
    for t in shuffled_t
        global T
        local Tt = tf.create_subtree(t,T)
        # print(Tt.z)
        local indices = tf.subtree_inputs(Tt,X,y)
        # println(indices)
        if !isempty(indices)
            # global priorT = T
            # println("prior T",T.branches)
            local Ttnew, better_found = optimize_node_parallel(Tt,indices,X,y,T)
            if better_found ==true
                global T = replace_subtree(T,Ttnew,X;print_prog=true)
            end
            # println("post T",T.branches)

            Lcur = loss(T,y)
            println("Lprev $Lprev, Lcur $Lcur")
            global tol = abs(Lprev - Lcur)
        end
    end
    println("Tolerance = $tol, Error = $Lcur, starting error = $starting_loss")
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
        # println("children",children)
        delete!(T.a,parent) # removed parent from branch
        delete!(T.b,parent)
        filter!(p -> p.first ∉ children, T.b)
        filter!(p -> p.first ∉ children, T.a)
        # println("parnet check",parent)
        filter!(x->x ≠parent,T.branches) # remove parent from branches
        filter!(x->x ∉children,T.nodes) # remove kid from nodes
        filter!(x->x ∉children,T.branches) # remove kid from branches
        filter!(x->x ∉children,T.leaves) # remove kid from leaves
        append!(T.leaves,parent) # add parent to leaves
        points = [k for (k,v) in T.z if v in children]
        for point in points # add assignments of z to parent
            T.z[point] = parent
            T.z[point] = parent
        end
        print
    else #
        parent = minimum(subtree.nodes)
        if print_prog == true
            println("replacement - branch $parent")
        end
        lower = parent *2
        upper = parent *2 + 1
        # println("parernt",parent)
        # lower_children = tf.get_children(lower,T)
        # upper_children = tf.get_children(upper,T)
        children = tf.get_children(parent,T)
        # println("children",children)
        # println("before T-nodes",T.nodes)
        if isnothing(children)==false
            filter!(x->x ∉ children,T.nodes) # remove kid from nodes
            filter!(x->x ∉ children,T.branches) # remove kid from branches
            filter!(x->x ∉ children,T.leaves) # remove kid from leaves
            filter!(p -> p.first ∉ children, T.b)
            filter!(p -> p.first ∉ children, T.a)
        end
        # println("after T-nodes",T.nodes)

        sub_children = tf.get_children(parent,subtree) # subtree children
        append!(T.nodes,sub_children)
        append!(T.leaves,sub_children)
        if parent ∉ T.branches
            append!(T.branches,parent)
        end
        T.a[parent] = subtree.a[parent]
        T.b[parent] = subtree.b[parent]
        filter!(x -> x ≠ parent, T.leaves) # incase replacing on leaf
        for point in keys(subtree.z) # add assignments of z to parent
            T.z[point] = subtree.z[point]
        end
    end
    return(T)
end

#-----
function loss(T,y;α=0.0001)
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
    # return (L/L̂ + α*C)*10000 #so that it not so small for now
    # return(L + α*C*L̂)
end

# TODO Can reduce this as there are no more indices ... tecnically
function _get_baseline(Y)
    Nt = sum((Y),dims=1)
    error = size(Y,1) - maximum(Nt)
    return error
end

function new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
    # println("test- in parallel split")
    branches = [root]
    leaves = [2*root,2*root+1]
    nodes = [root,2*root,2*root+1]
    n,p = size(XI)
    # Tfeas =
    Tfeas = tf.Tree(nodes,branches,leaves,Dict(),Dict(),Dict())

    new_values, error_best = _get_split(root,XI,YI,Tfeas,e,indices,X,y;Nmin=5)
    Tpara = tf.copy(Tfeas)
    Tpara.a[root] = Int(new_values[1])
    Tpara.b[root] = new_values[2]
    filter!(x->x≠root, Tpara.leaves)
    # println(Tfeas)
    Tfeas = tf.assign_class(X,Tpara,e;indices = indices)
    # println("Tpara z",Tpara.z)

    return Tpara
    # println("----newnodes2 check ",root,Tfeas)
    # Infeasible = true
    # count = 0
    # # while Infeasible
    # count+=1
    # # if count >1000
    # #     println("LIKELY INFEASIBLE -- $count")
    # # end
    # for j in 1:p
    #     values = sort(XI[:,j])
    #     # println("before - XI",sort(XI[:,j]))
    #     # local Tfeas
    #     for i in 1:n-1
    #         bsplit = 0.5*(values[i] + values[i+1])
    #         Tfeas.a[root] = j
    #         Tfeas.b[root] = bsplit
    #         # println("indices",indices)
    #         # println("bsplits ",bsplit)
    #         # println("----newnodes2 check ",root,Tfeas)
    #         Tfeas = tf.assign_class(X,Tfeas,e;indices = indices)
    #         # println("bsplitsafter  ",Tfeas.b)
    #         minsize = tf.minleafsize(Tfeas)
    #         # println("minssize",minsize)
    #         # println("--after--newnodes2  ",Tfeas.nodes)
    #         if (minsize >= Nmin)
    #             # println("minimum leaf size",minsize)
    #             # println(Tfeas)
    #             # global checkT = tf.copy(Tfeas)
    #             Infeasible = false
    #             # println("----newnodes2 after ",Tfeas.b)
    #             return Tfeas
    #         end
    #     end
    # end
    # if Infeasible == true
    #     return false
    # end
end

#--- Optimize Node Parallel
# Input: Subtree T to optimize, training data X,y
# Output: Subtree T with optimized parallel split at root
function optimize_node_parallel(Tt,indices,X,y,T)
    # println("Before",Tt)
    Y =tf.y_mat(y)
    # println(X)
    XI = X[indices,:]
    YI = Y[indices,:]
    root = minimum(Tt.nodes)
    e = tf.get_e(size(x,2))
    Tnew = Tt
    Tbest = Tt

    better_split_found = false

    error_best = loss(Tt,y)
    println("(Node $root)")
    # println(length(indices))
    # println("Error :",error_best)
    if root in Tt.branches #if the subtree is a branch of the full tree get its children
        println("Optimize-Node-Parallel : Subtree split")
        Tnew = tf.create_subtree(root,Tt)
        # println(Tnew)
        Tl = tf.left_child(root,Tt)
        Tu = tf.right_child(root,Tt)
        Tlower_sub = tf.create_subtree(Tl,Tt)
        Tupper_sub = tf.create_subtree(Tu,Tt)
        Tlower = replace_subtree(Tnew,Tlower_sub,X)
        Tupper = replace_subtree(Tnew,Tupper_sub,X)
        global Tlower_sub_check = Tlower_sub
        # global Ttest = Tlower
    else #it is a leaf -> create new leaf nodes
        println("Optimize-Node-Parallel : Leaf split")
        global Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
        if Tnew == false
            println("Optimize-Node-Parallerl: no feasible split found")
            return Tt
        end
        Tl = tf.left_child(root,Tnew)
        Tu = tf.right_child(root,Tnew)
        # println("bs, expect $root,",Tnew.b)
        # Tlnewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tlnew])
        # Tunewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tunew])
        # println("New tree ",Tnew)
        # println("Node lower ",Tl)
        # println("Node upper ",Tu)
        # println("new lower z ",Tln)
        # println("new upper z ",Tlnewz)
        # Tlower_sub = tf.Tree([Tlnew],[],[Tlnew],Tnew.a,Tnew.b,Tlnewz)
        # Tupper_sub = tf.Tree([Tunew],[],[Tunew],Tnew.a,Tnew.b,Tunewz)
        global Tlower_sub = tf.create_subtree(Tl,Tnew)
        Tupper_sub = tf.create_subtree(Tu,Tnew)
        Tlower = replace_subtree(Tnew,Tlower_sub,X)
        Tupper = replace_subtree(Tnew,Tupper_sub,X)
    end
    # println("last chcke?",length(Tlower.z),"   ",length(Tnew.z))

    # Uses Tnew to find best splits to add for the old tree
    Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y)

    # end
    # println(Tpara)
    # println("Errorbest before",error_best)
    if error_para < error_best
        println("!! Better Split Found : parallel")
        println("-->Error : $error_best => $error_para")
        Tt,error_best = Tpara,error_para
        better_split_found = true
    end
    # println(Tt)
    # if length(tf.subtree_inputs(Tlower,X,y)) > 0
    error_lower = loss(Tlower,y)
    if error_lower < error_best
        println("!! Better Split Found : lower")
        println("-->Error : $error_best => $error_lower")
        Tt,error_best = Tlower,error_lower
        better_split_found = true

        # delete!(Tt.a,root)
        # delete!(Tt.b,root)
    end
    error_upper = loss(Tupper,y)
    # end
    # println("Tt before",Tt.z)
    # if length(tf.subtree_inputs(Tupper,X,y)) > 0
    if error_upper < error_best
        println("!! Better Split Found : upper")
        println("-->Error : $error_best => $error_upper")
        Tt,error_best = Tupper,error_upper
        better_split_found = true

        # delete!(Tt.a,root)
        # delete!(Tt.b,root)
    end
    # println("Errorbe/st after",error_best)

    # end
    # println("Tt after",Tt.z)
    # println("After",Tt)
    return(Tt,better_split_found)
    end
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
    Tttry = tf.copy(Tt) #tf.Tree(Tt.nodes,Tt.branches,Tt.leaves,Tt.a,Tt.b,Tt.z)
    # append!(Tttry.nodes,[root*2,root*2+1])
    filter!(x->x≠root, Tttry.leaves)
    # append!(Tttry.leaves,[root*2,root*2+1])
    local new_values, error_best
    for j in 1:p, i in 1:n-1
        values = sort(XI[:,j])
        bsplit = 0.5*(values[i] + values[i+1])
        Tttry.a[root] = j
        Tttry.b[root] = bsplit
        Tttry = tf.assign_class(X,Tttry,e;indices = indices)
        #create a tree with this new a and b
        length_new =  length(tf.subtree_inputs(Tttry,X,y))
        error = loss(Tttry,y)
        if tf.minleafsize(Tttry) >= Nmin && length_new > 0 && error < error_best && true== true
            # print(i)
            error_best = error
            new_values = [j,bsplit]
            # println(j," ",bsplit)
        else
        end
    end
    return(new_values,error_best)
end

#---- Testing
root = 2
indices_test = tf.subtree_inputs(Tt_test,x,y)
XI_test = X[indices_test,:]
YI_test = tf.y_mat(y)[indices_test,:]
Tt_test = new_feasiblesplit(root,XI_test,YI_test,Tt_test,test_e,indices_test,X;Nmin=5)
tf.assign_class(x,Tt_test,test_e)


# T_test= tf.warm_start(tdepth,y,X,seed)
# test_e = tf.get_e(p)
# Tt_test = tf.create_subtree(root,T_test)
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
loss(T,y)
# Tt = tf.create_subtree(3,T)
Tt = tf.create_subtree(2,T)
# indices = tf.subtree_inputs(Tt,X,y)
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


# T
#
# bigkids = []
# global node_test = 3
# while node_test*2 ∈ T_old.nodes
#     global node_test
#     node_test = node_test*2
#     append!(bigkids,node_test)
# end
# bigkids
