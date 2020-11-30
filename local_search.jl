include("./tree.jl")
#input starting decision tree, training data X,y
#output locally optimal decision tree
using CSV, DataFrames, Random
cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")

depth = 2
T = tf.get_tree(depth)
iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]#[1:3,:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

#--- 8.1 Local Search

#function to calculate loss
function loss(T,X,y):
    for leaves in T.leaves
    end
end

#While loss is still decreasing
error_prev = loss(T,X,y)
tol = 1e-4
while abs(error_prev - error_cur) > tol #while improvements are still possible
    #Randomize the nodes
    shuffled_t = shuffle(T.nodes)
    for t in shuffled_t
        #get the indices of the nodes in the subtree and create Tree struct
        I = tf.nodes_subtree(2,T)
        Tt = tf.create_subtree(I,T)

        #Optimize subtree
        #XI =
        #YI =
        #Tt = optimize_node_parallel(Tt,XI,YI)

        #Replace the original tree with optimal subtree
        Tt = tf.replace_subtree(T,Tt)
    end
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
        Tl, Tu = new leaf nodes
    end
    error_best = loss(T,X,y)

    #T_para, error_para =
    ######
end

#--- Best Parallel Split
# Input: Tl and Tu subtrees as children of new split, X and y
# Output: Subtree with best parallel split at root, error of best tree

function minleafsize(T::Tree)
    ###
end

n = size(X,1)
p = size(X,2)
error_best = Inf
for j in 1:p
    values = sort(X[:,j])
    for i in 1:n-1
        b = 0.5*(values[i] + values[i+1])
        # T = branch node....
        if minleafsize(T) >= Nmin
            error = loss(T,X,y)
            if error < error_best
                error_best = error
                Tbest = T
            end
        end
    end
    return(Tbest, error_best)
end
