using Profile, BenchmarkTools
cd("C:/Users/Shaun Gan/Desktop/parallel-node-search/")
# cd("/Users/arkiratanglertsumpun/Documents/GitHub/parallel-node-search")
include("tree.jl")
include("local_search.jl")
include("local_search_z.jl")
include("local_search_half.jl")
include("unit_test.jl")

function splitobs(x,y,pct_train)
    xtrain = x[1:Int(floor(pct_train*size(x)[1])),:]
    xvalid = x[Int(floor(pct_train*size(x)[1]))+1:end,:]
    ytrain = y[1:Int(floor(pct_train*size(x)[1]))]
    yvalid = y[Int(floor(pct_train*size(x)[1]))+1:end]
    return(xtrain,xvalid,ytrain,yvalid)
end


iris_full = CSV.read("iris.csv",DataFrame)
iris = iris_full[randperm(size(iris_full,1)),:]
x = Matrix(iris[:,1:4])
y = Vector(iris[:,5])

xtrain,xvalid,ytrain,yvalid=splitobs(x,y,0.8)

#---------------
lend_full = CSV.read("../lending-club/lend_training_70.csv",DataFrame)
lend = lend_full[randperm(size(lend_full,1)),:][1:500,:]
x = Matrix(select(lend,Not(:loan_status)))
y = Vector(lend[:,:loan_status])
#split into training and validation sets
xtrain,xvalid,ytrain,yvalid = splitobs(x,y,0.8)
#------- Tuning
tune(1,xtrain,ytrain,xvalid,yvalid;nthreads = length(Sys.cpu_info()) -1 )
LocalSearch(x,y,3,400,α=0.01,tol_limit=0.1)
LocalSearch_z(x,y,2,400,α=0.01,numthreads=4,tol_limit=0.1)
LocalSearch(x,y,3,400,α=0.01,deep=true,tol_limit=0.1)
LocalSearch_half(x,y,2,400,α=0.01,numthreads=4,tol_limit=0.1)

#----- Profiling

@elapsed LocalSearch(x,y,1,400,α=0.01)
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
