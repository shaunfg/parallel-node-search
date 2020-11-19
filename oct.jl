include("./tree.jl")
import .tree
using JuMP, Gurobi

#----
"""
Nₖₜ: # of label k in node t

"""

function y_mat(y)
    n = length(y)
    y_class = int(categorical(y),type=Int)
    Y = zeros(n,k)
    for i in 1:n, k in y_class
        if y_class[i] == k
            Y[i,k] = 1
        end
    end
    return(Y)
end

tf.get_parent()
tf.get_tree()
#----
n = 1 #
K = 1 # number of labels k
p = 1 # number of rows of observed x
N_min = 1

tree = tf.get_tree(1)
model = Model(Gurobi.Optimizer)

@variable(model,z[1:n,tree.leaves],Bin)
@variable(model,l[tree.leaves],Bin)
@variable(model,c[1:K,tree.leaves],Bin)
@variable(model,a[1:p,tree.branches],Bin)
@variable(model,d[tree.branches],Bin)
@variable(model,b[tree.branches],Bin)


@constraint(model,[t=tree.leaves,k=1:K],Lₜ[t] ≥ Nₜ[t] - Nₖₜ[k,t] - n*(1-cₖₜ[k,t]))
@constraint(model,[t=tree.leaves,k=1:K],Lₜ[t] ≤ Nₜ[t] - Nₖₜ[k,t] + n*cₖₜ[k,t])
@constraint(model,[t=tree.leaves],Lₜ[t] ≥ 0)

#TODO _______
@constraint(model,[t=tree.leaves,k=1:K],Nₖₜ[k,t] = sum(z[i,t])) # for i:y_i=k

@constraint(model,[t=tree.leaves],Nₜ[t] = sum(z[:,t]))
@constraint(model,[t=tree.leaves],sum(c[:,t]) = l[t])
#TODO _______
@constraint(model,[i=1:n,t=tree.leaves,m]

@constraint(model,[i=1:n],sum(z[i,:]) = 1)
@constraint(model,[i=1:n,t=tree.leaves],z[i,t] ≤ l[t])
@constraint(model,[t=tree.leaves],sum(z[:,t]) ≥ N_min.*l[t])
@constraint(model,[t=tree.branches],sum(a[:,t]) = d[t])
@constraint(model,[t=tree.branches],0≤b[t]≤d[t])
#TODO _______
@constraint(model,[t=tree.branches[2:end],p=tf.get_parent(t)],d[t] <= d[p])

C = sum(d[t] for t in tree.branches)
