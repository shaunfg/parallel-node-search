using Profile, BenchmarkTools, DecisionTree
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
# cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")
include("tree.jl")
include("local_search.jl")
# LocalSearch(x,y,3,400,α=0.01,deep=true,tol_limit=0.1)

include("local_search_z.jl")
include("local_search_half.jl")
# include("local_search_deep.jl")
include("unit_test.jl")
include("model_evaluation.jl")

function splitobs(x,y,pct_train)
    xtrain = x[1:Int(floor(pct_train*size(x)[1])),:]
    xvalid = x[Int(floor(pct_train*size(x)[1]))+1:end,:]
    ytrain = y[1:Int(floor(pct_train*size(x)[1]))]
    yvalid = y[Int(floor(pct_train*size(x)[1]))+1:end]
    return(xtrain,xvalid,ytrain,yvalid)
end


pct = 0.7
n_obs = 500
iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
# iris = iris[1:Int(ceil(n_obs/pct)),:]
x_full = Matrix(iris[:,1:4])
y_full = Vector(iris[:,5])
x,xtest,y,ytest = splitobs(x_full,y_full,pct)

#xtrain,xvalid,ytrain,yvalid=splitobs(x,y,0.8)



#split into training and validation sets
#xtrain,xvalid,ytrain,yvalid = splitobs(x,y,0.8)
#------- Tuning
#tuned_results = tune(5,xtrain,ytrain,xvalid,yvalid;nthreads = length(Sys.cpu_info()) -8 )


#---------------

lend_full = CSV.read("../lending-club/lend_training_70.csv",DataFrame)
pct = 0.7
n_obs = 150
lend_full = filter!(x->x.loan_status!="late",lend_full)[1:Int(ceil(n_obs/pct)),:]
# lend = lend_full[randperm(size(lend_full,1)),:][1:200,:]
x_full = Matrix(select(lend_full,Not(:loan_status)))
y_full = Vector(lend_full[:,:loan_status])
x,xtest,y,ytest = splitobs(x_full,y_full,pct)
#testing dataset
# lend = lend_full[randperm(size(lend_full,1)),:][201:243,:]
# xtest = Matrix(select(lend,Not(:loan_status)))
# ytest = Vector(lend[:,:loan_status])

#----- Model Evaluation
nrestarts = 20
tdepth = 5
n_threads = 10
tol_limit = 1e-4
α = 0.1
seed_values = collect(100:100:100*nrestarts)
# #----- JIT COMPILE FIRST
# LocalSearch(x,y,3,400,α=0.01,tol_limit=0.001)
# LocalSearch_z(x,y,2,400,α=0.01,numthreads=4,tol_limit=0.1)
# LocalSearch(x,y,3,400,α=0.01,deep=true,tol_limit=0.1)
# LocalSearch_half(x,y,2,400,α=0.01,n_threads=4,tol_limit=0.1)
#
# serial_restarts!(x,y,nrestarts,tdepth,tol_limit = tol_limit,α=α)
# threaded_restarts!(x,y,nrestarts,tdepth,seed_values,n_threads,tol_limit = tol_limit,α=α)
# serial_half!(x,y,nrestarts,tdepth,α=α,tol_limit = tol_limit)
# serial_deep!(x,y,nrestarts,tdepth,α=α,tol_limit = tol_limit)
# serial_z!(x,y,nrestarts,tdepth;α=α,tol_limit = tol_limit,n_threads=n_threads)

#----- TIMINGSSSS
#
# p = size(x)[2]
# seed = 100
# rf_ntrees = 500
# rf = build_forest(y,x,floor(Int,sqrt(p)),rf_ntrees,0.7,tdepth,5,2,rng = seed)
# ypred = apply_forest(rf,xtest)
# acc = sum(ypred.==ytest)/length(ytest)

#serial
@elapsed T_serial = serial_restarts!(x,y,nrestarts,tdepth,tol_limit = tol_limit,α=α)
T_serial_f,serial_is_accuracy,serial_os_accuracy = evaluate_tree(T_serial,x,y,xtest,ytest)

#threaded
@elapsed T_threaded = threaded_restarts!(x,y,nrestarts,tdepth,seed_values,n_threads,tol_limit = tol_limit,α=α)
T_threaded_f,threaded_is_accuracy,threaded_os_accuracy = evaluate_tree(T_threaded,x,y,xtest,ytest)

#half
@elapsed T_half = serial_half!(x,y,nrestarts,tdepth,α=α,tol_limit = tol_limit)
T_half_f,half_is_accuracy,half_os_accuracy = evaluate_tree(T_half,x,y,xtest,ytest)

#deep !! REMEMBER TO CHANGE NUM THREAD IN .jl file
@elapsed T_deep = serial_deep!(x,y,nrestarts,tdepth,α=α,tol_limit = tol_limit)
T_deep_f,deep_is_accuracy,deep_os_accuracy = evaluate_tree(T_deep,x,y,xtest,ytest)

#z
@elapsed T_z = serial_z!(x,y,nrestarts,tdepth;α=α,tol_limit = tol_limit,n_threads=n_threads)
T_z_f,z_is_accuracy,z_os_accuracy = evaluate_tree(T_z,x,y,xtest,ytest)



@elapsed T_local = LocalSearch(x,y,4,100,α=0.0001)
@elapsed LocalSearch_z(x,y,2,400,α=0.01,numthreads=4)
@elapsed LocalSearch_z(x,y,2,100,α=0.01,numthreads=4)
#------
@profile LocalSearch(x,y,1,400,α=0.01)
Juno.profiler()
LocalSearch(x,y,1,400,α=0.01)
@profile LocalSearch_z(x,y,1,100,α=0.01,numthreads=4)
Juno.profiler()

@elapsed LocalSearch_z(x,y,1,100,α=0.01,numthreads=4)
@profile LocalSearch_z(x,y,1,100,α=0.01,numthreads=10)
Juno.profiler()

#-------------
LocalSearch(x,y,1,100,α=0.01)
# T_output = LocalSearch(xtrain,ytrain,10,100,α=0.01)
function func()
    LocalSearch(x,y,1,100,α=0.01)
end
time = @btime func()


# df = df[shuffle(1:size(df, 1)),:]


# loss(T,y,0.1)
# dmax = 8
# tune(dmax,x,y;nthreads = length(Sys.cpu_info()) -1 )
#
# trees = threaded_restarts!(x,y,nrestarts;warmup=400)
# ncores = length(Sys.cpu_info());
# Threads.nthreads()
#
# #parallelize random restarts
# nrestarts = 8
