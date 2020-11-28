include("./tree.jl")
# import .tree
using JuMP, Gurobi, CSV,Random

#------
iris_full = CSV.read("iris.csv")
iris = iris_full[randperm(size(iris_full,1)),:]#[1:3,:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

#----
# heart = CSV.read("heart.csv")
# # iris = iris_full[randperm(size(iris_full,1)),:]#[1:10,:]
# x = Matrix(heart[:,1:end-1])
# y = Vector(heart[:,end])
#----
n = size(x,1) #
p = size(x,2) # number of features
K = length(unique(y)) # number of labels k
N_min = 5
depth = 2
α = 0.1

ϵ = [minimum(abs.([x[i+1,j]-x[i,j] for i=1:size(x,1)-1 if x[i+1,j]!=x[i,j]]))
    for j=1:size(x,2)]
# ϵ = 0.1
# ϵmax = 0.1
ϵ = 1e-4

tree = tf.get_tree(depth)
model = Model(Gurobi.Optimizer)

@variable(model,z[1:n,tree.leaves],Bin)
@variable(model,l[tree.leaves],Bin)
@variable(model,c[1:K,tree.leaves],Bin)
@variable(model,d[tree.branches],Bin) #
@variable(model,a[1:p,tree.branches],Bin)

@variable(model,Nt[tree.leaves]≥ 0)
@variable(model,Nkt[1:K,tree.leaves]≥ 0)
@variable(model,Lt[tree.leaves]≥ 0)
@variable(model,b[tree.branches]≥ 0)

# integer_relationship_constraints
@constraint(model,[t=tree.leaves],sum(c[:,t]) == l[t]) # OK
@constraint(model,[i=1:n,t=tree.leaves],z[i,t] ≤ l[t])
@constraint(model,[i=1:n],sum(z[i,:]) == 1)
@constraint(model,[t=tree.leaves],sum(z[:,t]) ≥ N_min*l[t])
@constraint(model,[t=tree.branches[2:end],p=tf.get_parent(t)],d[t] ≤ d[p])
@constraint(model,[t=tree.branches],sum(a[:,t]) == d[t])

# leaf_samples_constraints
@constraint(model,[t=tree.leaves], Nt[t] == sum(z[:,t])) #
@constraint(model,[k=1:K,t=tree.leaves],Nkt[k,t] == sum(z[i,t]*tf.y_mat(y)[i,k] for i=1:n))#(1 + tf.y_mat(y)[i,k])/2 for i=1:n))

# leaf_error_constraints
@constraint(model,[t=tree.leaves,k=1:K],Lt[t] ≥ Nt[t] - Nkt[k,t] - n*(1-c[k,t]))
@constraint(model,[t=tree.leaves,k=1:K],Lt[t] ≤ Nt[t] - Nkt[k,t] + n*c[k,t])

# parent_branching_constraints
@constraint(model,[t=tree.branches],b[t]≤d[t])

# @constraint(model,[i=1:n,t=tree.leaves,m=tf.R(t)], a[:,m]'*x[i,:] ≥ b[m]-(1+ϵmax)*(1-z[i,t]))
# @constraint(model,[i=1:n,t=tree.leaves,m=tf.L(t)], a[:,m]'*x[i,:].+ϵ <= b[m] + (1+ϵmax)*(1-z[i,t]))

M = 2
@constraint(model,[i=1:n,t=tree.leaves,m=tf.R(t)],
            sum(a[j,m]*x[i,j] for j=1:p) ≥ b[m]-(1-z[i,t])*M)
@constraint(model,[i=1:n,t=tree.leaves,m=tf.L(t)],
            sum(a[j,m]*x[i,j] for j=1:p) + ϵ ≤ b[m] + (M+ϵ)*(1-z[i,t]))



# count = 0
# values = []
# for i=1:n
#     for t in tree.leaves
#         for m in tf.R(t)
#             println([m])
#             # println([i,t,m])
#             append!(values,[[i,t,m]])
#             global count +=1
#         end
#     end
# end
# values
# count

L_baseline = 100
@objective(model, Min, sum(Lt) + α*sum(d))

optimize!(model)
getobjectivevalue(model)
value.(Lt)
value.(a)
value.(b)
value.(Nt) # number of points in node t
value.(Nkt) # number of label k in node t
value.(d)
value.(c) # prediction label of each node
value.(l) # =1 if node leaf t contains any points
value.(z) # points in leaf node t
