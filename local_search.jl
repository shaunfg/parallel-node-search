include("./tree.jl")
#input starting decision tree, training data X,y
#output locally optimal decision tree

depth = 2
T = tf.get_tree(depth)
iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]#[1:3,:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

#--- 8.1 Local Search

#function to calculate loss
# function Loss(T,X,y):
#     for leaves in T.leaves
#     end
# end

#While loss is still decreasing

#Randomize the nodes
shuffled_t = shuffle(T.nodes)
for t in shuffled_t
    #get the indices of the nodes in the subtree and create Tree struct
    I = tf.nodes_subtree(2,T)
    Tt = tf.create_subtree(I,T)

    #Optimize subtree
    #To code in 8.2

    #Replace the original tree with optimal subtree
    #call on function: tf.replace_subtree
end

#--- Optimize Node Parallel


#--- Best Parallel Split
