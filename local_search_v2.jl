include("./tree_ls.jl")
#input starting decision tree, training data X,y
#output locally optimal decision tree
using CSV, DataFrames, Random, LinearAlgebra

cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")

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
e = 1*Matrix(I,p,p) #Identity matrix
#--- 8.1 Local Search

#random restart: initialize a random tree
seed = 100
Random.seed!(seed)
depth = 2
T = tf.get_randtree(depth,3)
m = length(T.branches) #num branches
z = zeros(n,length(T.nodes)) #observations x leaf assignments
m = length(T.branches)
b = rand(m,1) #randomize split thresholds for each branch
a = zeros(p,m) #features being splitted in each branch
for j in 1:m
    i = rand(1:p)
    a[i,j] = 1
end

#starting decision tree and assigments
tf.assign_class(x,T,a,b,z,e)
L = loss(T,Y,z)

#function to calculate loss
function loss(T,Y,z)
    L̂ = length(T.leaves)
    classes = size(Y,2) #number of classes
    Nt = zeros(length(T.nodes))
    Nkt = zeros(classes,length(T.nodes))
    Lt = zeros(length(T.nodes))
    L = 0
    for t in T.leaves
        Nt[t] = sum(z[:,t])
        for k in 1:classes
            Nkt[k,t] = z[:,t]'*Y[:,k]
        end
        Lt[t] = Nt[t] - maximum(Nkt[:,t])
    end
    L = sum(Lt)/L̂ #### + α*Cp......
    return(L)
end

#Initialize losses
Lprev = loss(T,Y,z)
tol = 100

while tol > 1e-4 #while improvements are still possible
    #Randomize the nodes
    shuffled_t = shuffle(T.nodes)
    for t in shuffled_t

        #get the indices of the nodes in the subtree and create subTree struct
        I = tf.nodes_subtree(2,T)
        Tt = tf.create_subtree(I,T)

        #Optimize subtree
        #XI =
        #YI =
        #Tt = optimize_node_parallel(Tt,XI,YI)

        #Replace the original tree with optimal subtree
        T = tf.replace_subtree(T,Tt)
        tf.assign_class(x,T,a,b,z,e)
    end
    Lcur = loss(T,Y,z)
    tol = abs(Lprev - Lcur)
end

#return T

#--- Optimize Node Parallel
# Input: Subtree T to optimize, training data X,y
# Output: Subtree T with optimized parallel split at root
function optimize_node_parallel(T::Tree,XI,YI)
    root = minimum(T.nodes)
    if root = Tt.branches #if the subtree is a branch get its children
        Tl, Tu = left_child(root,T), right_child(root,T)
    else #it is a leaf -> create new leaf nodes
        Tprog = progenate(root,T)
        Tl = left_child(root,Tprog)
        Tu = right_child(root,Tprog)
        #enforce splits for new branch
    end
    Lbest = loss(T,X,y)

    #Tpara, Lpara =
    ######
end

#--- Best Parallel Split
# Input: Tl and Tu subtrees as children of new split, X and y
# Output: Subtree with best parallel split at root, error of best tree

#Output the minimum number of observations in any leaf
function minleafsize(T,Nt)
    minbucket = Inf
    for t in T.leaves
        if Nt[t] < minbucket
            minbucket = Nt[t]
        end
    end
    return(minbucket)
end

function set_split(T,b)


n,p = size(X)
Lbest = Inf
for j in 1:p
    values = sort(X[:,j])
    for i in 1:n-1
        b = 0.5*(values[i] + values[i+1])
        T = assign_class

        if minleafsize(T,Nt) >= Nmin
            error = loss(T,X,y)
            if L < Lbest
                Lbest = L
                Tbest = T
            end
        end
    end
    return(Tbest, error_best)
end
