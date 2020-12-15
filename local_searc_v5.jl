using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
include("tree_ls_v2.jl")

iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

# Data pre-processing - Normalisation


#-------------
tdepth = 4
seed = 100

dt = fit(UnitRangeTransform, x, dims=1)
X = StatsBase.transform(dt,x)

T= tf.warm_start(tdepth,y,X,seed)
tol = 10
println("------------- NEW RUN ----------------")
# while tol >1e-5
for i =1:1
    # println("current $i")
    Lprev = loss(T,y)
    shuffled_t = T.nodes #shuffle(T.nodes)
    println("NODES", T.nodes)

    # for t in shuffled_t
    for t in [1]
        global T
        # println(T.branches)
        # println(T.b)
        local Tt = tf.create_subtree(t,T)
        # print(Tt.z)
        local indices = tf.subtree_inputs(Tt,X,y)
        # println(indices)
        if !isempty(indices)
            # println("----- Tree before",Tt)
            println("bsss",T.b)
            local Ttnew = optimize_node_parallel(Tt,indices,X,y)
            # println(Ttnew)
            # println("--------------", Ttnew)
            # if Ttnew != false
            # printl
            global T = replace_subtree(T,Ttnew,X)
            # println("NODES", T)
            # println("NEW B",T.b)
            # end
            # println("newtree",T)
            Lcur = loss(T,y)
            println("Lprev $Lprev, Lcur $Lcur")
            global tol = abs(Lprev - Lcur)
        end
    end
    println("Tolerance = $tol, Error = $Lprev")
end

testdict
function replace_subtree(T,subtree,X)#::Tree,subtree::Tree
    if length(subtree.nodes) == 1 #leaf
        kid = minimum(subtree.nodes)
        println("REPLACING LEAF ",kid)
        if mod(kid,2) == 1
            # branches have no zs, dont need to worry
            parent = tf.get_parent(kid)
            # sibling = tf.left_child(parent,T)
        else
            parent = tf.get_parent(kid)
            # sibling = tf.right_child(parent,T)
        end
        children = tf.get_children(parent,T)
        println("children",children)
        delete!(T.a,parent) # removed parent from branch
        delete!(T.b,parent)
        filter!(p -> p.first ∉ children, T.b)
        filter!(p -> p.first ∉ children, T.a)
        filter!(x->x ≠parent,T.branches) # remove parent from branches
        # filter!(x->x ≠sibling,T.nodes) # remove sibling from nodes
        # filter!(x->x ≠sibling,T.branches) # remove siblingfrom branches
        # filter!(x->x ≠sibling,T.leaves) # remove siblingfrom branches
        # filter!(x->x ≠kid,T.nodes) # remove kid from nodes
        # filter!(x->x ≠kid,T.branches) # remove kid from branches
        # filter!(x->x ≠kid,T.leaves) # remove kid from branches
        # filter!(x->x ≠parent,T.branches) # remove parent from branches
        filter!(x->x ∉children,T.nodes) # remove kid from nodes
        filter!(x->x ∉children,T.branches) # remove kid from branches
        filter!(x->x ∉children,T.leaves) # remove kid from leaves
        append!(T.leaves,parent) # add parent to leaves
        points = [k for (k,v) in T.z if v in children]
        for point in points # add assignments of z to parent
            T.z[point] = parent
            T.z[point] = parent
        end
    else # branch
        parent = minimum(subtree.nodes)
        lower = parent *2
        upper = parent *2 + 1
        println("parernt",parent)
        # lower_children = tf.get_children(lower,T)
        # upper_children = tf.get_children(upper,T)
        children = tf.get_children(parent,T)
        println("children",children)

        filter!(x->x ∉ children,T.nodes) # remove kid from nodes
        filter!(x->x ∉ children,T.branches) # remove kid from branches
        filter!(x->x ∉ children,T.leaves) # remove kid from leaves
        filter!(p -> p.first ∉ children, T.b)
        filter!(p -> p.first ∉ children, T.a)

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
function loss(T,y;α=1)
    Y_full = tf.y_mat(y)
    L̂ = _get_baseline(Y_full)
    z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
    Y = Y_full[z_keys,:] # Reorganize Y values to z order
    z_values = collect(values(T.z)) #
    Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ T.leaves]
    Nt = length.([[k for (k,v) in T.z if v ==t] for t ∈ T.leaves])
    Lt = [Nt[t] - maximum(Nkt[t]) for t=1:length(T.leaves)]
    L = sum(Lt)/L̂ #### + α*Cp......
    C = sum(T.branches)
    return(L/L̂ + α*C)
end

# TODO Can reduce this as there are no more indices ... tecnically
function _get_baseline(Y)
    Nt = sum((Y),dims=1)
    error = size(Y,1) - maximum(Nt)
    return error
end

function new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
    # println("test- in parallel split")
    branches = root
    leaves = [2*root,2*root+1]
    nodes = [root,2*root,2*root+1]
    n,p = size(XI)
    Tfeas = tf.Tree(nodes,branches,leaves,Dict(),Dict(),Dict())
    # println("----newnodes2 check ",root,Tfeas)
    Infeasible = true
    count = 0
    # while Infeasible
    count+=1
    if count >1000
        println("LIKELY INFEASIBLE -- $count")
    end
    for j in 1:p
        values = sort(XI[:,j])
        # println("before - XI",sort(XI[:,j]))
        for i in 1:n-1
            bsplit = 0.5*(values[i] + values[i+1])
            Tfeas.a[root] = j
            Tfeas.b[root] = bsplit
            # println("----newnodes2 check ",root,Tfeas)
            Tfeas = tf.assign_class(X,Tfeas,e;indices = indices)
            minsize = tf.minleafsize(Tt)
            # println("--after--newnodes2  ",Tfeas.nodes)
            if (minsize >= Nmin)
                Infeasible = false
                # println("----newnodes2 after ",Tfeas.b)
                return Tfeas
            end
        end
    end
    if Infeasible == true
        throw(DomainError("No are feasible split available.. truncating"))
    end
end

#--- Optimize Node Parallel
# Input: Subtree T to optimize, training data X,y
# Output: Subtree T with optimized parallel split at root
function optimize_node_parallel(Tt,indices,X,y)
    Y =tf.y_mat(y)
    # println(X)
    XI = X[indices,:]
    YI = Y[indices,:]
    root = minimum(Tt.nodes)
    e = tf.get_e(size(x,2))
    Tnew = Tt
    Tbest = Tt

    error_best = loss(Tt,y)
    println("---------$root-----------")
    # println("Error :",error_best)
    if root in Tt.branches #if the subtree is a branch of the full tree get its children
        println("New subtree")
        Tnew = tf.create_subtree(root,Tt)
        Tl = tf.left_child(root,Tt)
        Tu = tf.right_child(root,Tt)
        Tlower = tf.create_subtree(Tl,Tt)
        Tupper = tf.create_subtree(Tu,Tt)
        println(Tlower)
        global Ttest = Tlower
    else #it is a leaf -> create new leaf nodes
        println("New leaf")
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
        Tlnew = tf.left_child(root,Tnew)
        Tunew = tf.right_child(root,Tnew)
        # println("bs, expect $root,",Tnew.b)
        Tlnewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tlnew])
        Tunewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tunew])
        Tlower = tf.Tree([Tlnew],[],[Tlnew],Tnew.a,Tnew.b,Tlnewz)
        Tupper = tf.Tree([Tunew],[],[Tunew],Tnew.a,Tnew.b,Tunewz)
    end
    # Uses Tnew to find best splits to add for the old tree
    Tpara, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y)

    # end
    # println(Tpara)

    if error_para < error_best
        println("replacement-parallel")
        println("replacement-parallel")
        println("Error : $error_best => $error_para")
        Tt,error_best = Tpara,error_para
    end
    # println(Tt)
    if length(tf.subtree_inputs(Tlower,X,y)) > 0
        error_lower = loss(Tlower,y)
        if error_lower < error_best
            println("replacement-lower")
            println("replacement-lower")
            println("Error : $error_best => $error_lower")
            Tt,error_best = Tlower,error_lower
            # delete!(Tt.a,root)
            # delete!(Tt.b,root)
        end
        error_upper = loss(Tupper,y)
    end
    # println("Tt before",Tt.z)
    if length(tf.subtree_inputs(Tupper,X,y)) > 0
        if error_upper < error_best
            println("replacement-upper")
            println("replacement-upper")
            println("Error : $error_best => $error_upper")
            Tt,error_best = Tupper,error_upper
            # delete!(Tt.a,root)
            # delete!(Tt.b,root)
        end
    end
    # println("Tt after",Tt.z)
    return(Tt)
end


function best_parallelsplit(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    #println("test- in parallel split")
    new_values, error_best = _get_split(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    Tpara = tf.copy(Tt)
    Tpara.a[root] = Int(new_values[1])
    Tpara.b[root] = new_values[2]
    filter!(x->x≠root, Tpara.leaves)
    # Tpara = tf.assign_class(X,Tpara,e)
    # replace_subtree(T,Ttnew,X)
    # println()
    return(Tpara,error_best)
end

function _get_split(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    n,p = size(XI)
    error_best = Inf
    Tttry = tf.Tree(Tt.nodes,Tt.branches,Tt.leaves,Tt.a,Tt.b,Tt.z)
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


T= tf.warm_start(4,y,X,seed)
# Tt = tf.create_subtree(3,T)
Tt = tf.create_subtree(7,T)
indices = tf.subtree_inputs(Tt,X,y)
Ttnew = optimize_node_parallel(Tt,indices,X,y)

T_old = tf.Tree(Any[1, 2,3,6,7,12,13,14,15,28,29],
                Any[1,3,6,7,14],
                Any[2, 12, 13,15,28,29],
                Dict{Any,Any}(6=> 2,3 => 4,1 => 3,7=>1,14 =>1),
                Dict{Any,Any}(6 => 0.5,3 => 0.6458333333333333,1 => 0.24576271186440676,7=>0.5,14 =>0.5),
                Dict(10 => 2,85 => 13, 20 => 12,1=>15,2=>28,40=>29))


output = replace_subtree(T_old,Ttnew,X)
unique(values(output.z))
output.leaves
output.a

(loss(T,y),length(T.z))
# (loss(Tt,y),length(Tt.z))
(loss(Ttnew,y),length(Tt.z))
T_combined = replace_subtree(T,Ttnew,X)
(loss(T_combined,y),length(T_combined.z))
