using CSV, DataFrames, StatsBase, CategoricalArrays, Random
include("tree.jl")

function one_hot_encode(X, names)
    X2 = deepcopy(X)
    deletecols!(X2, Symbol.(names))
    for i in names
        vales = unique(X[i])
        for j in vales
           X2[Symbol(string(j))] = (X[i].==j)*1
        end
    end
    return X2
end

accepted_full = CSV.read("../lending-club/accepted_2007_to_2018Q4.csv")

vars = [:id,:loan_amnt,:funded_amnt,:term,:int_rate,
        :installment,:grade,:emp_length,
        :home_ownership,:annual_inc,:verification_status,
        :loan_status,:pymnt_plan,:purpose,:title,
        :tot_hi_cred_lim,:total_bal_ex_mort,:total_bc_limit,
        :total_il_high_credit_limit,:dti,:delinq_2yrs,
        :fico_range_low,:fico_range_high]

accepted_id = select(accepted_full,vars)
accepted = select(accepted_id,Not(:id))
description = describe(accepted)

many_missing = findall(description[:,:nmissing] .> 100)
check = describe(accepted[:,many_missing])

drop_cols = names(accepted[:,many_missing])
new_vars = [x for x in vars if x ∉ Symbol.(drop_cols) ]

output = select(accepted_full,new_vars)
output = output[completecases(output),:]
check2 = describe(output)
[names(output) check2[:,:eltype]]
# output[:,:term] =

new_df = DataFrame()
# one_hot_encode(new_df,[:loan_status])
new_df[:,:term] = [x == " 36 months" ? 36 : 60 for x in output[:,:term]]
new_df[:,:grade] = [x ∈ ["A","B","C"] ? 1 : 0 for x in output[:,:grade]]
new_df[:,:pymnt_plan] = [x == "y" ? 1 : 0 for x in output[:,:pymnt_plan]]
new_df[:,:verification_status] = [x ∈ ["Verified","Source Verified"] ? 1 : 0
                for x in output[:,:verification_status]]

new_df[:,:purpose] .= ""
for (i,x) in enumerate(output[:,:purpose])
    if x ∈ ["debt_consolidation","credit_card"]
            new_df[i,:purpose] = "credit"
    elseif x ∈ ["small_business","home_improvement","major_purchase","house",
                "vacation","car","medical","moving","renewable_energy","wedding",
                "educational"]
            new_df[i,:purpose] = "big-purchase"
    else
            new_df[i,:purpose] = "other"
    end
end
new_df = one_hot_encode(new_df,[:purpose])

new_df[:,:loan_status] .= ""
for (i,x) in enumerate(output[:,:loan_status])
        if x ∈ [ "Does not meet the credit policy. Status:Charged Off","Charged Off","Default"]
                new_df[i,:loan_status] = "not-paid"
        elseif x ∈ [ "In Grace Period","Late (31-120 days)","Late (16-30 days)"]
                new_df[i,:loan_status] = "late"
        elseif x ∈ ["Current"]
                new_df[i,:loan_status] = "current"
        else
                new_df[i,:loan_status] = "paid"
        end
end

flots = select(output,[:id,:loan_amnt,:funded_amnt,:int_rate,:installment,
                       :annual_inc,:delinq_2yrs,:fico_range_low,:fico_range_high])
final_df = hcat(new_df,flots)

useful = filter(row -> row.loan_status != "current",final_df)

shuffled_useful = useful[shuffle(axes(useful, 1)), :]
trainidx = Int(floor(nrow(useful)*0.7))

training = shuffled_useful[1:trainidx,:]
testing = shuffled_useful[trainidx+1:end,:]

CSV.write("../lending-club/lend_training_70.csv",training)
CSV.write("../lending-club/lend_testing_30.csv",testing)

using Test
@test nrow(training)+ nrow(testing)== nrow(useful)
# training = filter(row -> row.loan_status != "current",new_df)
# y1 = [x ∈ ["OTHER","ANY"] ? "OTHER" : x for x in output[:,:home_ownership]]
# y1_hot = tf.y_mat(y1)
