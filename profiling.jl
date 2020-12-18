using CSV, DataFrames, Random, LinearAlgebra, Distributions, StatsBase
using BenchmarkTools
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
# cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")
include("tree.jl")

lend_full = CSV.read("../lending-club/lend_training_70.csv",DataFrame)
lend = lend_full[randperm(size(lend_full,1)),:][1:500,:]
x = Matrix(select(lend,Not(:loan_status)))
y = Vector(lend[:,:loan_status])
#split into training and validation sets

dt = fit(UnitRangeTransform, x, dims=1)
X = StatsBase.transform(dt,x)
# X = X[:,!Nan]
e = tf.get_e(size(x,2))
T = tf.warm_start(10,y,X,100)
function preprocess_loss(y,T)
    z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
    Y_full = tf.y_mat(y)
    Y = Y_full[z_keys,:]
    L̂ = _get_baseline(Y_full)
    return (Y,L̂)
end

function _get_baseline(Y)
    Nt = sum((Y),dims=1)
    error = size(Y,1) - maximum(Nt)
    return error
end

@elapsed loss_better(T,boo[1],0.01,boo[2])
@elapsed loss(T,y,0.01)

boo = preprocess_loss(y,T)
@btime for i=1:100 loss_better(T,boo[1],0.01,boo[2])end
@btime for i =1:100 loss(T,y,0.01) end

using Profile
@profile loss(T,y,0.01)
@profile (for i=1:100 loss_better(T,boo[1],0.01,boo[2])end)
Juno.profiler()

function loss_better(T,Y,α,L̂)
    # z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
    # z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
    # Y = Y_full[z_keys,:] # Reorganize Y values to z order
    Nkt = [sum((Y[values(T.z) .== t,:]),dims=1) for t ∈ T.leaves]
    Nt = length.([[k for (k,v) in T.z if v ==t] for t ∈ T.leaves])
    Lt = [Nt[t] - maximum(Nkt[t]) for t=1:length(T.leaves)]
    L = sum(Lt)#/L̂ #### + α*Cp......
    C = length(T.branches)
    # println("L̂ = $L̂,L = $L, C = $(α*C)")
    # return (L) #so that it not so small for now
    # println(α*C)
    return (L/L̂*1 + α*C) #so that it not so small for now
end

@inline function loss(T,y,α)
    Y_full = tf.y_mat(y)
    L̂ = _get_baseline(Y_full)
    z_keys = collect(keys(T.z)) # ONLY CHECK points in subtree
    Y = Y_full[z_keys,:] # Reorganize Y values to z order
    z_values = collect(values(T.z)) #
    @inbounds Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ T.leaves]
    @inbounds Nt = length.([[k for (k,v) in T.z if v ==t] for t ∈ T.leaves])
    @inbounds Lt = [Nt[t] - maximum(Nkt[t]) for t=1:length(T.leaves)]
    @inbounds L = sum(Lt)#/L̂ #### + α*Cp......
    @inbounds C = length(T.branches)
    # println("L̂ = $L̂,L = $L, C = $(α*C)")
    # return (L) #so that it not so small for now
    return (L/L̂*1 + α*C) #so that it not so small for now
end
