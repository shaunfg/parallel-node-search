using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
include("./tree_ls_v2.jl")
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")


iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]#[1:3,:]
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
tdepth = 3
seed = 100
T= tf.warm_start(tdepth,y,x,seed)

for i=1:100
    Lprev = loss(T,Y)
end

function loss(T,y)
    L̂ = length(T.leaves)
    z_keys = collect(keys(T.z))
    Y = tf.y_mat(y)[z_keys,:] # Reorganize Y values to z order
    z_values = collect(values(T.z)) #
    Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ T.leaves]
    Nt = length.([[k for (k,v) in T.z if v ==t] for t ∈ T.leaves])
    Lt = [Nt[t] - maximum(Nkt[t]) for t=1:length(T.leaves)]
    L = sum(Lt)/L̂ #### + α*Cp......
    return(L)
end

loss(T,y)

tol
# while tol > 1
for i=1:1
    Lprev = loss(T,y)
    shuffled_t = T.nodes
    for t in shuffled_t
        Tt = tf.create_subtree(t,T)
        indices,XI,YI = tf.subtree_inputs(Tt,x,y)
        Tt = optimize_node_parallel
    end
end


test = Dict("A"=> 3, "B" => 3, "C" => 2)
leaves = [3,2]

Dict(k => test[k] for (k,v) in test if v in leaves)

# [k for k in keys(test) if values(test) in leaves]
[x for x in values(test)if x in leaves]
Dict(key => test[key] for key in branches)
