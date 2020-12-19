using Plots,CSV, DataFrames

df = CSV.read("results.csv")

# df = df[completecases(df),:]

df = DataFrame(df)

df1= df[((df[:,:cp].==0.0001) .& (df[:,:MaxDepth].==4)), :]
bar(df1.Model,df1.out_sample)
ylims!(0.6,0.9)


df2= df[((df[:,:cp].==0.0001) .& (df[:,:MaxDepth].==4) .&(df[:,:Train].==200)), :]
df2 = df2[Not(df2.Model .=="OCT MIO"),:]
bar(df2.Model,df2.out_sample)
bar(df2.Model,df2.seconds,ylabel="Seconds",label = "Model",legend=false)#,title = "α=0.0001, Max Depth = 4, N_train = 200")

df3= df[((df[:,:Model].=="Half") .&(df[:,:Dataset] .=="Lending Club")),:]# .& (df[:,:MaxDepth].==4) .&(df[:,:Train].==200)), :]

bar(df2.Model,df2.seconds,ylabel="Seconds")#,title = "α=0.0001, Max Depth = 4, N_train = 200")

y1 = [563.5
      31.24
      25
      16.75
      38.19
      134.15]

y2 = [54.34
        5.62
        5.06
        3.05
        9.41
        169.15]

x = ["OCT MIO"
        "Serial"
        "Threaded Restarts"
        "Half"
        "Deep"
        "z"]

bar(x,y1)



ntrees = [100*50
        200*50
        500*20
        1000*10]

times = [16.75
        134.53
        164.62
        416.96]

plot(ntrees,times)


df4 = df[1:6,:]
bar(df4.Model,df4.seconds)
