#Problem Set 6 - Sushant Singh
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, DataFramesMeta
function q3e()
#1.
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body)
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
θ̂_glm = glm(@formula(Y ~ Odometer*RouteUsage*Branded*time + Odometer^2*RouteUsage^2*Branded*time^2), df_long, Binomial(), LogitLink())
println(θ̂_glm)
#----------------------------------------------
#3(a).
function create_grids()

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

#3(b).
Odometer=kron(ones(zbin),xval)
RouteUsage= kron(ones(xbin),zval)
Branded=zeros(size(Odometer))
time=zeros(size(Branded))

df1 = DataFrame(Odometer=Odometer,RouteUsage=RouteUsage,Branded=Branded,time=time)
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body)
Y = Matrix(df[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
X = Matrix(df[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
Z = Vector(df[:,:RouteUsage])
B = Vector(df[:,:Branded])
N = size(Y,1)
T = size(Y,2)
Xstate = Matrix(df[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
Zstate = Vector(df[:,:Zst])

FV=zeros(zbin*xbin,2,T+1)

for t= 2:T
    for b= 0:1
        for i=1:size(Branded)
        df1.time[i]=t
        df1.Branded[i]=b
        p0=predict(θ̂_glm,df1)
        row = x + (z-1)*xbin
        FV[row,b+1,t] =-β*log(p0)
        end
    end
end

FVT1=zeros(N,T)
for i= 1:N
    for t= 1:T
    row0 = (Zstate[i]-1)*xbin+1
    row1  = Xstate[i,t] + (Zstate[i]-1)*xbin 
    FVT1[i,t] = (xtran[row1,:].-xtran[row0,:])'*FV[row0:row0+xbin-1,B[i]+1,t+1]
    end
end

fvt1= FVT1'[:]

#3(c).
df_long = @transform(df_long,fv = fvt1)
theta_hat_ccp_glm = glm(@formula(Y ~ Odometer + Branded),df_long, Binomial(), LogitLink(),offset=df_long.fv)

#3(d)
end

#3(e)
@time q3e()