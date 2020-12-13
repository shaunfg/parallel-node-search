using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
include("tree_ls_v2.jl")

iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

# Data pre-processing - Normalisation
dt = fit(UnitRangeTransform, x, dims=1)
# x_copy = x
X = StatsBase.transform(dt,x)

#Initialize parameters
n = size(x,1) #num observations
p = size(x,2) #num features
#-------------
tdepth = 1
seed = 100
T= tf.warm_start(tdepth,y,x,seed)


loss(T,y)

tol = 1
# while tol > 1
for i=1:1
    Lprev = loss(T,y)
    shuffled_t = T.nodes
    for t in shuffled_t
        global T
        local Tt = tf.create_subtree(t,T)
        # print(Tt.z)
        local indices = tf.subtree_inputs(Tt,x,y)
        if !isempty(indices)
            println("----- Tree before",Tt)
            Ttnew = optimize_node_parallel(Tt,indices,X,y)
            # println("--------------", Ttnew)
            global T = tfreplace_subtree(T,Ttnew)
        end
        Lcur = loss(T,y)
        tol = abs(Lprev - Lcur)
    end
end

# t = 1
# Tt = tf.create_subtree(t,T)
# indices,XI,YI = tf.subtree_inputs(Tt,x,y)

#----
test_e = tf.get_e(p)
Tt_test = tf.create_subtree(3,T)
indices_test = tf.subtree_inputs(Tt_test,x,y)
XI_test = X[indices_test,:]
# YI_test = tf.y_mat(y)[indices_test,:]
# Tt_test = tf.progenate(4,Tt_test,test_e,XI_test,YI_test,indices_test,X;Nmin=5)
# tf.assign_class(x,Tt_test,test_e)
#
#
# Tt_out = new_feasiblesplit(4,XI_test,YI_test,Tt_test,test_e,indices_test,X;Nmin=5)


function loss(T,y)
    L̂ = length(T.leaves)
    z_keys = collect(keys(T.z))
    # println("check1",size(y))
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
            for i in 1:n-1
                bsplit = 0.5*(values[i] + values[i+1])
                Tfeas.a[root] = j
                Tfeas.b[root] = bsplit
                println("----newnodes2 check ",root,Tfeas)
                Tfeas = tf.assign_class(X,Tfeas,e;indices = indices)
                minsize = tf.minleafsize(Tt)
                # println("--after--newnodes2  ",Tfeas.nodes)
                if (minsize >= Nmin)
                    Infeasible = false
                    # println("----newnodes2 after ",Tfeas)
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
    zbest = tf.assign_class(X,Tt,e,indices = indices)
    Tnew = Tt
    Tbest = Tt

    abest = copy(Tt.a)
    bbest = copy(Tt.b)
    #println(minleafsize(Tt,zbest))
    # print(size(y))
    # println("check1",size(y))
    error_best = loss(Tt,y)
    println("Error :",error_best)
    if root in Tt.branches #if the subtree is a branch of the full tree get its children
        Tl = tf.left_child(root,Tt)
        Tu = tf.right_child(root,Tt)
        Tlower = tf.create_subtree(Tl,Tt)
        Tupper = tf.create_subtree(Tu,Tt)
    else #it is a leaf -> create new leaf nodes
        Tnew  = new_feasiblesplit(root,XI,YI,Tt,e,indices,X;Nmin=5)
        Tlnew = tf.left_child(root,Tnew)
        Tunew = tf.right_child(root,Tnew)
        Tlnewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tlnew])
        Tunewz = Dict(k => T.z[k] for (k,v) in T.z if v in [Tunew])
        Tlower = tf.Tree([Tlnew],[],[Tlnew],Tnew.a,Tnew.b,Tlnewz)
        Tupper = tf.Tree([Tunew],[],[Tunew],Tnew.a,Tnew.b,Tunewz)
    end
    Tpara, error_para= best_parallelsplit(root,XI,YI,Tnew,e,indices,X,y)
    # println(Tpara)
    # println(Tpara)
    if error_para < error_best
        println("replacement-parallel")
        println("Error para, Error best: ",error_para,", ",error_best)
        Tt,error_best = Tpara,error_para
    end
    # println(Tt)
    if length(tf.subtree_inputs(Tlower,X,y)) > 0
        error_lower = loss(Tlower,y)
        if error_lower < error_best
            println("replacement-lower")
            Tt,error_best = Tlower,error_lower
            delete!(Tt.a,root)
            delete!(Tt.b,root)
        end
        error_upper = loss(Tupper,y)
    end
    if length(tf.subtree_inputs(Tupper,X,y)) > 0
        if error_upper < error_best
            println("replacement-upper")
            Tt,error_best = Tupper,error_upper
            delete!(Tt.a,root)
            delete!(Tt.b,root)
        end
    end
    println(Tt)
    return(Tt)#,abest,bbest)
    # #println("Saving :", minleafsize(Tt,zbest))
end


function best_parallelsplit(root,XI,YI,Tt,e,indices,X,y;Nmin=5)
    #println("test- in parallel split")
    n,p = size(XI)
    error_best = Inf
    Tbest = Tt
    Tttry = Tt

    for j in 1:p
        values = sort(XI[:,j])
        for i in 1:n-1
            bsplit = 0.5*(values[i] + values[i+1])
            Tttry = tf.assign_class(X,Tt,e;indices = indices)
            Tttry.a[root] = j
            Tttry.b[root] = bsplit
            #create a tree with this new a and b
            if tf.minleafsize(Tttry) >= Nmin
                #println(minleafsize(Tt,ztry))
                error = loss(Tttry,y)
                #println("Best_error, Error: ",error_best,", ", error)
                if error < error_best
                    error_best = error
                    Tbest = Tttry
                end
            end
        end
    end
    return(Tbest,error_best)
end
