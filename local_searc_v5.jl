using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
include("tree_ls_v2.jl")

iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

# Data pre-processing - Normalisation
dt = fit(UnitRangeTransform, x, dims=1)
X = StatsBase.transform(dt,x)

#Initialize parameters
n = size(x,1) #num observations
p = size(x,2) #num features
#-------------
tdepth = 2
seed = 100

T= tf.warm_start(tdepth,y,X,seed)
tol = 10
println("------------- NEW RUN ----------------")
while tol >1e-5
    # println("current $i")
    Lprev = loss(T,y)
    shuffled_t = T.nodes
    for t in shuffled_t
        global T
        println(T.branches)
        println(T.b)
        local Tt = tf.create_subtree(t,T)
        # print(Tt.z)
        local indices = tf.subtree_inputs(Tt,X,y)
        # println(indices)
        if !isempty(indices)
            # println("----- Tree before",Tt)
            local Ttnew = optimize_node_parallel(Tt,indices,X,y)
            # println(Ttnew)
            # println("--------------", Ttnew)
            if Ttnew != false
                global T = replace_subtree(T,Ttnew,X,t)
            end
            # println("newtree",T)
            Lcur = loss(T,y)
            global tol = abs(Lprev - Lcur)
        end
    end
    println("Tolerance = $tol, Error = $Lprev")
end

function replace_subtree(T,subtree,X,t)#::Tree,subtree::Tree
    subtree_root = minimum(subtree.nodes)
    nodes_to_delete = tf._nodes_subtree(subtree_root,T)
    e = tf.get_e(size(X,2))
    new_leaves = T.leaves[(.!(in(nodes_to_delete).(T.leaves)))]
    new_branches = T.branches[(.!(in(nodes_to_delete).(T.branches)))]
    for j in subtree.nodes
        if j in subtree.leaves
            append!(new_leaves,j)
        else
            append!(new_branches,j)
        end
    end
    keep_nodes = unique(vcat(new_leaves,new_branches))
    println(new_leaves,new_branches)
    println("a valu---",subtree.a)
    for key in keys(subtree.a)
        T.a[key] = subtree.a[key]
        T.b[key] = subtree.b[key]
    end

    for point in keys(subtree.z)
        T.z[point] = subtree.z[point]
    end
    newtree = tf.Tree(keep_nodes,new_branches,new_leaves,T.a,T.b,T.z)
    return newtree#Tree(keep_nodes,new_branches,new_leaves,t.a,t.b,t.z)
end


#-----
function loss(T,y)
    L̂ = length(T.leaves)
    z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
    Y = tf.y_mat(y)[z_keys,:] # Reorganize Y values to z order
    z_values = collect(values(T.z)) #
    Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ T.leaves]
    Nt = length.([[k for (k,v) in T.z if v ==t] for t ∈ T.leaves])
    Lt = [Nt[t] - maximum(Nkt[t]) for t=1:length(T.leaves)]
    L = sum(Lt)/L̂ #### + α*Cp......
    return(L)
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
    while Infeasible
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
    else #it is a leaf -> create new leaf nodes
        println("New leaf")
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
        Tlnew = tf.left_child(root,Tnew)
        Tunew = tf.right_child(root,Tnew)
        println("bs, expect $root,",Tnew.b)
        Tlnewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tlnew])
        Tunewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tunew])
        Tlower = tf.Tree([Tlnew],[],[Tlnew],Tnew.a,Tnew.b,Tlnewz)
        Tupper = tf.Tree([Tunew],[],[Tunew],Tnew.a,Tnew.b,Tunewz)
    end
    # Uses Tnew to find best splits to add for the old tree
    splits, error_para = best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y)
    Tpara = tf.copy(Tt)
    if root in Tt.branches
        Tpara.a[root] = Int(splits[1])
        Tpara.b[root] = splits[2]
    end

    if error_para < error_best
        println("replacement-parallel")
        println("Error : $error_best => $error_para")
        Tt,error_best = Tpara,error_para
        return(Tt)
    end
    # println(Tt)
    if length(tf.subtree_inputs(Tlower,X,y)) > 0
        error_lower = loss(Tlower,y)
        if error_lower < error_best
            println("replacement-lower")
            println("Error : $error_best => $error_lower")
            Tt,error_best = Tlower,error_lower
            delete!(Tt.a,root)
            delete!(Tt.b,root)
            return(Tt)
        end
        error_upper = loss(Tupper,y)
    end
    if length(tf.subtree_inputs(Tupper,X,y)) > 0
        if error_upper < error_best
            println("replacement-upper")
            println("Error : $error_best => $error_upper")
            Tt,error_best = Tupper,error_upper
            delete!(Tt.a,root)
            delete!(Tt.b,root)
            return(Tt)
        end
    end
    return(false)
end


function best_parallelsplit(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    #println("test- in parallel split")
    n,p = size(XI)
    error_best = Inf
    Tttry = tf.Tree(Tt.nodes,Tt.branches,Tt.leaves,Tt.a,Tt.b,Tt.z)
    local new_values, error_best
    for j in 1:p, i in 1:n-1
        values = sort(XI[:,j])
        bsplit = 0.5*(values[i] + values[i+1])

        Tttry = tf.assign_class(X,Tt,e;indices = indices)
        Tttry.a[root] = j
        Tttry.b[root] = bsplit
        #create a tree with this new a and b
        length_new =  length(tf.subtree_inputs(Tttry,X,y))
        error = loss(Tttry,y)
        if tf.minleafsize(Tttry) >= Nmin && length_new > 0 && error < error_best && true== true
            # print(i)
            error_best = error
            new_values = [j,bsplit]
            println(j,bsplit)
        else
        end
    end
    return(new_values,error_best)
end
#---- Testing
root = 7
T_test= tf.warm_start(tdepth,y,X,seed)
test_e = tf.get_e(p)
Tt_test = tf.create_subtree(root,T_test)
indices_test = tf.subtree_inputs(Tt_test,x,y)
XI_test = X[indices_test,:]
YI_test = tf.y_mat(y)[indices_test,:]
Tt_test = new_feasiblesplit(root,XI_test,YI_test,Tt_test,test_e,indices_test,X;Nmin=5)
tf.assign_class(x,Tt_test,test_e)


Tnew  = new_feasiblesplit(root,XI_test,YI_test,Tt_test,test_e,indices_test,X;Nmin=5)
Tlnew = tf.left_child(root,Tnew)
Tunew = tf.right_child(root,Tnew)
println("bs, expect $root,",Tnew.b)
Tlower = tf.Tree([Tlnew],[],[Tlnew],Tnew.a,Tnew.b,Tnewz)
Tupper = tf.Tree([Tunew],[],[Tunew],Tnew.a,Tnew.b,Tnewz)
Tpara = best_parallelsplit(root,XI_test,YI_test,Tnew,test_e,indices_test,X,y)

# Tt_out = new_feasiblesplit(4,XI_test,YI_test,Tt_test,test_e,indices_test,X;Nmin=5)


test = true
if test != false
    print("hi")
end
