#Problem Set 2 - Sushant Singh
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
#1.
using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = [df.age df.white df.collgrad]
Z = hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4, df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8)
y = df.occupation
 function mlogit(Beta, X, Z,y)

    K = size(X,2)+1
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigBeta = [reshape(Beta,K,J-1) zeros(K)]

    bigZ = zeros(N,J)
    for j=1:J
        bigZ[:,j] = Z[:,j]-Z[:,J]
    end

    num = zeros(N,J)
    dem = zeros(N)
    for j=1:J
        XZ=cat(X,bigZ[:,j],dims=2)
        num[:,j] = exp.(XZ*bigBeta[:,j])
        dem .+= num[:,j]
    end
    P = num./repeat(dem,1,J)
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end
mlogit_hat_optim = optimize(b-> mlogit(b,X,Z, y), rand(7*(size(X,2)+1)), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(mlogit_hat_optim.minimizer)
#----------------------------------------------
#2.
println("γ - the estimated coefficient, can be interpreted as the impact of alternative specific difference on utility difference")
#----------------------------------------------
#3.
function nlogit(βwc,βbc,λwc,λbc, X, Z,y)

    K = size(X,2)+1
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)

    bigYwc = zeros(N,3)
    for j=1:3
        bigYwc[:,j] = y.==j
    end
    bigβwc = [reshape(βwc,K,3) zeros(K)]

    bigZwc = zeros(N,3)
    for j=1:3
        bigZwc[:,j] = Z[:,j]-Z[:,J]
    end

    numwc1 = zeros(N,3)
    demwc1 = zeros(N)
    numwc = zeros(N,3)
    for j=1:3
        XZwc=cat(X,bigZwc[:,j],dims=2)
        numwc1[:,j] = exp.((XZwc*bigβwc[:,j])./λwc)
        demwc1 .+= numwc1[:,j]
        demwc=(repeat(demwc1,1,3))^λwc
        numwc2=(repeat(demwc1,1,3))^(λwc-1)
        numwc[:,j]=numwc1[:,j].*numwc2
    end

    bigYbc = zeros(N,4)
    for j=1:4
        bigYbc[:,j] = y.==j+3
    end
    bigβbc = [reshape(βbc,K,4) zeros(K)]

    bigZbc = zeros(N,4)
    for j=1:4
        bigZbc[:,j] = Z[:,j+3]-Z[:,J]
    end

    numbc1 = zeros(N,4)
    dembc1 = zeros(N)
    numbc = zeros(N,4)
    for j=1:4
        XZbc=cat(X,bigZbc[:,j],dims=2)
        numbc1[:,j] = exp.((XZbc*bigβbc[:,j])./λbc)
        dembc1 .+= numbc1[:,j]
        dembc=(repeat(dembc1,1,4))^λbc
        numbc2=(repeat(dembc1,1,4))^(λbc-1)
        numbc[:,j]=numbc1[:,j].*numbc2
    end
    
    num=cat(numwc,numbc,dims=2)
    dem=1+demwc+dembc
    P = num./dem
    
    loglike = -sum( bigY.*log.(P) )
    
    return loglike
end

nlogit_hat_optim = optimize((βwc,βbc,λwc,λbc)->nlogit(βwc,βbc,λwc,λbc, X, Z,y), rand(7*(size(X,2)+1)+2), LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
println(nlogit_hat_optim.minimizer)

end

#----------------------------------------------
#4.
q4()
