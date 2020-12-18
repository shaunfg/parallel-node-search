function accuracy(fitted_T,xtest,Y_full,xtrain,ytrain)
    p = size(xtest,2) #num features
    e = tf.get_e(p)
    z_keys = collect(keys(fitted_T.z)) # ONLY CHECK points in subtree
    Y = ytrain[z_keys,:] # Reorganize Y values to z order
    z_values = collect(values(fitted_T.z)) #
    # println(Y)
    Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ fitted_T.leaves]
    pred = zeros(size(Nkt))
    for t in 1:length(Nkt)
        rowmax = maximum(Nkt[t])
        pred[t] = findmax(Nkt[t])[2][2]
    end

    # prediction
    pred_T = tf.Tree(deepcopy(fitted_T.nodes),deepcopy(fitted_T.branches),
                deepcopy(fitted_T.leaves),deepcopy(fitted_T.a),
                deepcopy(fitted_T.b),Dict())
    pred_T = tf.assign_class(xtest,pred_T,e;indices = false)

    z_keys = collect(keys(pred_T.z)) # ONLY CHECK points in subtree
    Y = Y_full[z_keys,:] # Reorganize Y values to z order
    z_values = collect(values(pred_T.z)) #
    println(size(Y),size(Y_full),unique(collect(values(pred_T.z))))
    Nkt = [sum((Y[z_values .== t,:]),dims=1) for t ∈ pred_T.leaves]


    Nt = length.([[k for (k,v) in pred_T.z if v ==t] for t ∈ pred_T.leaves])
    Trues = 0.0
    for t in 1:length(Nkt)
        # println(pred[t])
        # println(Nkt[t])
        Trues += Nkt[t][Int(pred[t])]
    end
    accuracy = Trues/sum(Nt)
    return(accuracy)
end

function evaluate_tree(T_list,x,y,xtest,ytest)
    best_accuracy = 0
    best_T = []
    is_accuracy = 0
    for t in 1:length(T_list)
        is_accuracy = accuracy(T_list[t],x,tf.y_mat(y),x,tf.y_mat(y))
        #println(t, ": ",is_accuracy)
        if is_accuracy > best_accuracy
            best_T = T_list[t]
            best_accuracy = is_accuracy
            #println("Best accuracy: ",best_accuracy)
        end
    end
    os_accuracy = accuracy(best_T,xtest,tf.y_mat(ytest),x,tf.y_mat(y))
    return(best_T,best_accuracy,os_accuracy)
end
