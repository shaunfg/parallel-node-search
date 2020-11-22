include("./tree.jl")
# import .tree
using JuMP, Gurobi, CSV,Random

iris_full = CSV.read("iris.csv")
iris = iris_full[randperm(size(iris_full,1)),:]#[1:10,:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

#----
n = size(x,1) #
K = length(unique(y)) # number of labels k
p = size(x,2) # number of features
N_min = 1
depth = 3

ϵ = 1e5
ϵmax = 1e5
α = 1

tree = tf.get_tree(depth)
model = Model(Gurobi.Optimizer)

@variable(model,z[1:n,tree.leaves],Bin)
@variable(model,l[tree.leaves],Bin)
@variable(model,c[1:K,tree.leaves],Bin)
@variable(model,a[1:p,tree.branches],Bin)
@variable(model,d[tree.branches],Bin) #
@variable(model,b[tree.branches],Bin)
@variable(model,Lt[tree.leaves])

@variable(model,C)
@variable(model,Nt[tree.leaves])
@variable(model,Nkt[1:K,tree.leaves])

@constraint(model,[t=tree.leaves,k=1:K],Lt[t] ≥ Nt[t] - Nkt[k,t] - n*(1-c[k,t]))
@constraint(model,[t=tree.leaves,k=1:K],Lt[t] ≤ Nt[t] - Nkt[k,t] + n*c[k,t])
@constraint(model,[t=tree.leaves],Lt[t] ≥ 0) # OK

@constraint(model,[k=1:K,t=tree.leaves],Nkt[k,t] == sum(tf.y_mat(y)[i,k]*z[i,t] for i=1:n))
@constraint(model,[t=tree.leaves], Nt[t] == sum(z[:,t]))
@constraint(model,[t=tree.leaves],sum(c[:,t]) == l[t])
@constraint(model,C == sum(d))

@constraint(model,[i=1:n, t=tree.leaves, m=tf.R(t)], a[:,m]' * x[i,:] >= b[m] - (1 -z[i,t]))
@constraint(model,[i=1:n,t=tree.leaves,m=tf.L(t)], a[:,m]'*(x[i,:].+ϵ) <= b[m] + (1+ϵmax)*(1-z[i,t]))

@constraint(model,[i=1:n],sum(z[i,:]) == 1)
@constraint(model,[i=1:n,t=tree.leaves],z[i,t] ≤ l[t])
@constraint(model,[t=tree.leaves],sum(z[:,t]) ≥ N_min*l[t])
@constraint(model,[t=tree.branches],sum(a[:,t]) == d[t])
@constraint(model,[t=tree.branches],b[t]≤d[t])
@constraint(model,[t=tree.branches],b[t]≥0)
@constraint(model,[t=tree.branches[2:end],p=tf.get_parent(t)],d[t] <= d[p])

L_baseline = 100
@objective(model, Min, 1/L_baseline * sum(Lt) + α*C)

optimize!(model)
value.(a)
value.(b)
zs = value.(z)
