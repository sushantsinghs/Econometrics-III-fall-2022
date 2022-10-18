#Problem Set 7 - Sushant Singh 
#Discussed with Junyeol Ryu
using SMM
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using CSV

function q6()
#1.
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function gmm(beta, X, y)
    g = y .-X*beta
    J = g'*I*g
    return J
end
optim_gmm = optimize(beta ->gmm(beta, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
println(optim_gmm.minimizer)

beta_ols = inv(X'*X)*X'*y
println(beta_ols)
#----------------------------------------------
#2(a).
freqtable(df, :occupation) 
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation)

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function logit_ml(alpha, X, y)    
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
    num = zeros(N,J)
    dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
    P = num./repeat(dem,1,J)
        
    loglike = -sum( bigY.*log.(P) )
        
    return loglike
end
alpha_true = [.1910213,-.0335262,.5963968,.4165052,-.1698368,-.0359784,1.30684,-.430997,.6894727,-.0104578,.5231634,-1.492475,-2.26748,-.0053001,1.391402,-.9849661,-1.398468,-.0142969,-.0176531,-1.495123,.2454891,-.0067267,-.5382892,-3.78975]
alpha_start = alpha_true.*rand(size(alpha_true))
println(size(alpha_true))
alpha_hat_optim = optimize(a -> logit_ml(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
optim_ml = alpha_hat_optim.minimizer
println(optim_ml)

#2(b).
function logit_gmm(alpha, X, y)
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
        
    num = zeros(N,J)
    dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
    P = num./repeat(dem,1,J)
    g = bigY .- P
    gmm = sum(g'*I*g)
    return gmm
end
optim_logit_gmm_b = optimize(a -> logit_gmm(a, X, y), optim_ml, LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
println(optim_logit_gmm_b.minimizer)

#2(c)
optim_logit_gmm_c = optimize(a -> logit_gmm(a, X, y), alpha_start, LBFGS(), Optim.Options(g_tol=1e-8, iterations=100_000))
println(optim_logit_gmm_c.minimizer)
#The estimates from 2(b) and 2(c) is not the same. So, the objective function is globally concave.

#----------------------------------------------
#3.
function simulate()
    K = 2
    N = 10000
    J = 3
    X=rand(N,J)
    β=[1 2;3 4]
    num = zeros(N,J)
    dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*β[:,j])
            dem .+= num[:,j]
        end
        
    P = num./repeat(dem,1,J)
    ε = randn(N,1)
    Y=zeros(N,1)
       for i= 1: N
        Y[i,1]=sum(sum(P[i,:])>ε[i,1]) + sum((P[i,2]+P[i,3])>ε[i,1]) + sum(P[i,3]>ε[i,1])
       end
    end
simulate()
#----------------------------------------------
#5.
function mlogit_smm(θ, X, y, D)
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
        for j=1:J
            bigY[:,j] = y.==j
        end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    β = θ[1:end-1]
    σ = θ[end]
    if length(β)==1
        β = β[1]
    end  
    num = zeros(N,J)
    dem = zeros(N)
        for j=1:J
            num[:,j] = exp.(X*bigAlpha[:,j])
            dem .+= num[:,j]
        end
        
    P = num./repeat(dem,1,J)

    gmodel = zeros(N+1,J,D)
    gdata  = vcat(bigY,var(bigY))

    Random.seed!(1234)  
    for d=1:D
        ε = σ*randn(N,J)
        ỹ = P .+ ε
        gmodel[1:end-1,1:J,d] = ỹ
        gmodel[  end  ,1:J,d] = var(ỹ)
    end

    err = vec(gdata .- mean(gmodel; dims=2))

    J = err'*I*err
    return J
end
#----------------------------------------------
#6.
println(q6())
