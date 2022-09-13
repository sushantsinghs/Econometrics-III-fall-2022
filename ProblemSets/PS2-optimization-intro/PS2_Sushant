#Problem Set 2 - Sushant Singh
#using Pkg
#Pkg.add("Optim")
#Pkg.add("HTTP")
#Pkg.add("GLM")

using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
function q(6)
#1.
f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
minusf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1)   
result = optimize(minusf, startval, BFGS())
println(result)
#----------------------------------------------
#2.
using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

bols = inv(X'*X)*X'*y
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println(bols_lm)
#----------------------------------------------
#3.
function logit(alpha, X, d)
    loglike=d'*X*alpha-sum(log.(ones(size(X,1),1)+exp.(X*alpha)))
    return loglike
end
alpha_hat_logit = optimize(a -> -(logit(a, X, y)), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(alpha_hat_logit.minimizer)
#----------------------------------------------
#4. 
alpha_glm= glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
println(alpha_glm)
#----------------------------------------------
#5.
freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==8 ,:occupation] .= 7
df[df.occupation.==9 ,:occupation] .= 7
df[df.occupation.==10,:occupation] .= 7
df[df.occupation.==11,:occupation] .= 7
df[df.occupation.==12,:occupation] .= 7
df[df.occupation.==13,:occupation] .= 7
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

y1=y.==1
y2=y.==2
y3=y.==3
y4=y.==4
y5=y.==5
y6=y.==6
y7=y.==7

Y = cat(y1,y2,y3,y4,y5,y6,dims=2) 

function csum(A)
    Colsum = []
    for i=1:size(A,1)
        sum(A[i,:])
        push!(Colsum,sum(A[i,:]))
    end
        return Colsum
end

function mlogit(alpha, X, d)
    loglike=tr(d*(X*alpha)')-sum(log.(csum(exp.(X*alpha))))
    return loglike
end

beta_hat_mlogit1 = optimize(b -> -mlogit(b, X, Y), zeros(size(X,2),6), LBFGS(), Optim.Options(g_tol=10-5, iterations=100_000, show_trace=true))
println(beta_hat_mlogit1.minimizer)

beta_hat_mlogit2 = optimize(b -> -mlogit(b, X, Y), rand(size(X,2),6), LBFGS(), Optim.Options(g_tol=10-5, iterations=100_000, show_trace=true))
println(beta_hat_mlogit2.minimizer)

beta_hat_mlogit3 = optimize(b -> -mlogit(b, X, Y), 2*rand(size(X,2),6).-1, LBFGS(), Optim.Options(g_tol=10-5, iterations=100_000, show_trace=true))
println(beta_hat_mlogit3.minimizer)

end
#----------------------------------------------
#6.
q(6)