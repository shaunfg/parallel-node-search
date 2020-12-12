include("./tree_ls_v2.jl")
#input starting decision tree, training data X,y
#output locally optimal decision tree
using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase

cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")
#cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")

iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]#[1:3,:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

#----
#Data pre-processing
dt = fit(UnitRangeTransform, x, dims=1)
x_copy = x
x = StatsBase.transform(dt,x)
Y = tf.y_mat(y) #matrix mapping observations and classes

#Initialize parameters
n = size(x,1) #num observations
p = size(x,2) #num features
#--- 8.1 Local Search

#random restart: initialize a random tree
seed = 100
tdepth = 2

#starting decision tree
T,a,b,z,e = tf.warm_start(tdepth,y,x,seed)

#function to calculate loss
#function to calculate loss
function loss(T,Y,z)
    L̂ = length(T.leaves)
    #Y_mat = tf.y_mat(Y)
    classes = size(Y,2) #number of classes
    Nt = zeros(maximum(T.nodes))
    Nkt = zeros(classes,maximum(T.nodes))
    Lt = zeros(maximum(T.nodes))
    L = 0
    z_mat = zeros(size(Y,1),maximum(T.nodes))
    for t in T.leaves
        i = [k for (k,v) in z if v==t]
        z_mat[i,t] .= 1
        Nt[t] = length(i)
        for k in 1:classes
            Nkt[k,t] = z_mat[:,t]'*Y[:,k]
        end
        Lt[t] = Nt[t] - maximum(Nkt[:,t])
    end
    L = sum(Lt)/L̂ #### + α*Cp......
    return(L)
end

function loss_subtree(Tt,YI,z,indices,Y)
    L̂ = length(Tt.leaves)
    #Y_mat = tf.y_mat(Y)
    classes = size(Y,2) #number of classes
    Nt = zeros(maximum(Tt.nodes))
    Nkt = zeros(classes,maximum(Tt.nodes))
    Lt = zeros(maximum(Tt.nodes))
    L = 0
    z_mat = zeros(maximum([k for (k,v) in z]),maximum(Tt.nodes))
    #print(z_mat)
    for t in Tt.leaves
        i = [k for (k,v) in z if v==t]
        z_mat[i,t] .= 1
        Nt[t] = sum(z_mat[:,t])
        for k in 1:classes
            for ii in i
                if Y[ii,k] == 1
                    Nkt[k,t] += z_mat[ii,t]
                end
            end
            #println(t,"Passed")
        end
        Lt[t] = Nt[t] - maximum(Nkt[:,t])
    end
    L = sum(Lt)/L̂ #### + α*Cp......
    return(L)
end

#Initialize losses

include("./tree_ls_v2.jl")


T,a,b,z,e = tf.warm_start(tdepth,y,x,seed)
iter = 0
tol = 100
while tol > 1e-4 #while improvements are still possible
    Lprev = loss(T,Y,z)
    #Randomize the nodes
    shuffled_t = shuffle(T.nodes)
    for t in 2#shuffled_t
        iter += 1
        println("Iter: ",iter)
        #println(T)
        #println("Reading :", minleafsize(T,z))
        #Create the subtree struct
        subtree_nodes = tf.nodes_subtree(t,T)
        Tt = tf.create_subtree(subtree_nodes,T)
        #Get the data and matrices for the nodes in the subtree
        indices,XI,YI = tf.subtree_inputs(Tt,z,x,Y)
        #Optimize subtree
        if !isempty(XI)
            #println(b)
            Ttnew,anew,bnew = optimize_node_parallel(Tt,XI,YI,a,b,z,indices,x,Y)
            #print(bnew)
            #global a = anew
            #global b = bnew
            #local Tt = Ttnew
            #Replace the original tree with optimal subtree
            T = tf.replace_subtree(T,Ttnew)
            #println(T)
            a = anew
            b = bnew
            z = tf.assign_class(x,T,a,b,e,z)
            #println("Saving :", minleafsize(T,z))
        end
        Lcur = loss(T,Y,z)
        tol = abs(Lprev - Lcur)
    end
    #print(tol)
end


#--- Optimize Node Parallel
# Input: Subtree T to optimize, training data X,y
# Output: Subtree T with optimized parallel split at root
function optimize_node_parallel(Tt,XI,YI,a,b,z,indices,X,Y)
    root = minimum(Tt.nodes)
    znew = copy(z)
    zbest = tf.assign_class_subtree(X,Tt,a,b,e,z,indices)
    abest = copy(a)
    bbest = copy(b)
    anew = copy(a)
    bnew = copy(b)
    #println(minleafsize(Tt,zbest))
    error_best = loss_subtree(Tt,YI,zbest,indices,Y)
    println("Error :",error_best)
    if root in Tt.branches #if the subtree is a branch of the full tree get its children
        #println("test-branch",root)
        Tl = tf.left_child(root,Tt)
        Tu = tf.right_child(root,Tt)
        Tlower = tf.create_subtree(tf.nodes_subtree(Tl,Tt),Tt)
        Tupper = tf.create_subtree(tf.nodes_subtree(Tu,Tt),Tt)
    else #it is a leaf -> create new leaf nodes
        #println("test-leaf",root)
        Tt = tf.progenate(root,Tt)
        #Tl and Tu are indices of lower and upper children
        Tlnew = tf.left_child(root,Tt)
        Tunew = tf.right_child(root,Tt)
        Tlower = tf.Tree([Tlnew],[],[Tlnew])
        Tupper = tf.Tree([Tunew],[],[Tunew])
        anew, bnew, znew = new_feasiblesplit(root,XI,YI,Tt,anew,bnew,e,znew,indices,X)
    end
    Tpara, error_para, apara, bpara, zpara = best_parallelsplit(root,XI,YI,Tt,anew,bnew,e,znew,indices,X,Y)
    if error_para <= error_best
        print("Error para, Error best: ",error_para,", ",error_best)
        Tt,error_best,abest,bbest,zbest = Tpara,error_para,apara,bpara,zpara
    end

    zlower = tf.assign_class_subtree(X,Tlower,anew,bnew,e,znew,indices)
    zupper = tf.assign_class_subtree(X,Tupper,anew,bnew,e,znew,indices)

    error_lower = loss_subtree(Tlower,YI,zlower,indices,Y)
    if error_lower < error_best
        Tt,error_best,abest,bbest,zbest = Tlower,error_lower,anew,bnew
        delete!(abest,root)
        delete!(bbest,root)
    end
    error_upper = loss_subtree(Tupper,YI,zupper,indices,Y)
    if error_upper < error_best
        Tt,error_best,abest,bbest,zbest = Tupper,error_upper,anew,bnew
        delete!(abest,root)
        delete!(bbest,root)
    end
    #println("Saving :", minleafsize(Tt,zbest))
    #z = tf.assign_class(X,Tt,abest,bbest,e,zbest)
    return(Tt,abest,bbest)
end

#--- Best Parallel Split
# Input: Tl and Tu subtrees as children of new split, X and y
# Output: Subtree with best parallel split at root, error of best tree

#Output the minimum number of observations in any leaf
function minleafsize(T,z)
    minbucket = Inf
    Nt = zeros(maximum(T.nodes))
    z_mat = zeros(maximum([k for (k,v) in z]),maximum(T.nodes))
    for t in T.leaves
        i = [k for (k,v) in z if v==t]
        z_mat[i,t] .= 1
        Nt[t] = sum(z_mat[:,t])
        if (Nt[t] > 0) & (Nt[t] < minbucket)
            minbucket = Nt[t]
        end
    end
    return(minbucket)
end

length([k for (k,v) in z if v==511])

function best_parallelsplit(root,XI,YI,Tt,anew,bnew,e,z,indices,X,Y;Nmin=5)
    #println("test- in parallel split")
    n,p = size(XI)
    error_best = Inf
    Tbest = Tt
    Tttry = Tt
    abest = copy(anew)
    bbest = copy(bnew)
    atry = copy(anew)
    btry = copy(bnew)
    zbest = copy(z)
    ztry = copy(z)
    #println(root,": ",minleafsize(Tt,z))
    for j in 1:p
        values = sort(XI[:,j])
        for i in 1:n-1
            bsplit = 0.5*(values[i] + values[i+1])
            atry[root] = j
            btry[root] = bsplit
            #create a tree with this new a and b
            #println("btry: ",btry)
            ztry = tf.assign_class_subtree(X,Tttry,atry,btry,e,ztry,indices)
            # println("test-assignedclass")
            # if root == 255
            # end
            if minleafsize(Tttry,ztry) >= Nmin
                #println(minleafsize(Tt,ztry))
                error = loss_subtree(Tttry,YI,ztry,indices,Y)
                #println("Best_error, Error: ",error_best,", ", error)
                if error < error_best
                    error_best = error
                    Tbest = Tttry
                    abest = atry
                    bbest = btry
                    zbest = ztry
                end
            end
        end
    end
    #println("bbest: ",bbest)
    #println("Inner minleaf: ",minleafsize(Tbest,zbest))
    #println("Inner bbest: ",bbest)
    return(Tbest, error_best, abest, bbest, zbest)
end


function new_feasiblesplit(root,XI,YI,Tt,anew,bnew,e,z,indices,X;Nmin=5)
    # println("test- in parallel split")
    n,p = size(XI)
    Tfeas = Tt
    atry = copy(anew)
    btry = copy(bnew)
    ztry = copy(z)
    Infeasible = true
    while Infeasible
        for j in 1:p
            values = sort(XI[:,j])
            for i in 1:n-1
                bsplit = 0.5*(values[i] + values[i+1])
                atry[root] = j
                btry[root] = bsplit
                ztry = tf.assign_class_subtree(X,Tt,atry,btry,e,ztry,indices)
                if (minleafsize(Tt,ztry) >= Nmin)
                    Infeasible = false
                    return(atry, btry, ztry)
                    end
                end
            end
        end
    end
end
