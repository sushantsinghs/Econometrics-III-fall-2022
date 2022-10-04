#Problem Set 5 - Sushant Singh
using Pkg
Pkg.add("DataFramesMeta")
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, DataFrames
function q3g()
#1.
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/
master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
df = @transform(df, bus_id = 1:size(df,1))
dfy = @select(df, :bus_id,:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20,:RouteUsage,:Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id,:RouteUsage,:Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfy_long, Not(:variable))
dfx = @select(df, :bus_id,:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, time = kron(collect([1:20]...),ones(size(df,1))))
select!(dfx_long, Not(:variable))
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id,:time])
sort!(df_long,[:bus_id,:time])

#----------------------------------------------
#2.
sigma_hat = glm(@formula(Y ~ Odometer + Branded ), df_long, Binomial(), LogitLink())
println(sigma_hat)
#----------------------------------------------
#3.
#(a)
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/blob/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body)
Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
X1 = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
X2 = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
#(b)
function xgrid(theta,xval)
    N      = length(xval)
    xub    = vcat(xval[2:N],Inf)
    xtran1  = zeros(N,N)
    xtran1c = zeros(N,N)
    lcdf   = zeros(N)
    for i=1:length(xval)
        xtran1[:,i]   = (xub[i] .>= xval).*(1 .- exp.(-theta*(xub[i] .- xval)) .- lcdf)
        lcdf        .+= xtran1[:,i]
        xtran1c[:,i] .+= lcdf
    end
    return xtran1, xtran1c
end
zval   = collect([.25:.01:1.25]...)
zbin   = length(zval)
xval   = collect([0:.125:25]...)
xbin   = length(xval)
tbin   = xbin*zbin
xtran  = zeros(tbin,xbin)
xtranc = zeros(xbin,xbin,xbin)
for z=1:zbin
    xtran[(z-1)*xbin+1:z*xbin,:],xtranc[:,:,z] = xgrid(zval[z],xval)
end
return zval,zbin,xval,xbin,xtran
end
zval,zbin,xval,xbin,xtran = create_grids()
@views @inbounds function q3e()
#(c)
T=20
future_value_array=zeros(size(xtran,1),2,T+1)
β=0.9
for t=T+1:1
  for b=1:size(df,1)
   for z=1:zbin
    for x=1:xbin
       row=xval[x] + (zval[z]-1)*xbin
       v1[t]= 1.92596-0.148154*xval[x]+1.05919*Branded[b]+xtran[row,:]'*FV[(zval[z]-1)*xbin+1:zval[z]*xbin,b+1,t+1]
       v0[t]= xtran[1+(zval[z]-1)*xbin,:]'*FV[(zval[z]-1)*xbin+1:zval[z]*xbin,b+1,t+1]
       futurevaluearray[:,:,t]=β*log(exp(v0[t])+exp(v1[t]))
end
end
end
end
#(d)
log_likelihood_value=zeros(T+1)
for t=T+1:1
row0=1+(:Zst-1)*xbin
row1=:Xst+(:Zst-1)*xbin
v1t-v0t 
(xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,B[i]+1,t+1]
P_i1t =exp(v1t −v0t)/(1+exp(v1t −v0t))
P_i0t = 1−P_i1t
log_likelihood_value[t]=d_ijt*log(P_ijt)
loglikehood=sum(log_likelihood_value)
end
return -loglikehood
end
end
#(e)
println(q3e())
#(f)
#-
#(g)
println(q3g())