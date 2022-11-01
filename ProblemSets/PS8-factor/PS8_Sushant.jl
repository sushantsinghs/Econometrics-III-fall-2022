#Problem Set 8 - Sushant Singh 

#using Pkg
#Pkg.add("MultivariateStats")
using Optim
using HTTP
using GLM
using LinearAlgebra
using Random
using Statistics
using DataFrames
using DataFramesMeta
using Distributions
using CSV
using MultivariateStats

function allwrap()
#1. 
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/blob/master/ProblemSets/PS8-factor/nlsy.csv"
df = CSV.read(HTTP.get(url).body)
ols_lm = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
println(ols_lm)

#----------------------------------------------
#2.
matrix_asvab = hcat(df.asvabAR, df.asvabCS,df.asvabMK,df.asvabNO,df.asvabPC,df.asvabWK)
cor_asvab = cor(matrix_asvab)
println(cor_asvab)

#----------------------------------------------
#3.
ols_asvab_lm = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabAR + asvabCS + asvabMK + asvabNO + asvabPC + asvabWK), df)
println(ols_asvab_lm)
#Yes, it will be problematic to directly include these in the regression, since the correlation between these varibales is high. 
 
#----------------------------------------------
#4.
asvabMat = (hcat(df.asvabAR, df.asvabCS,df.asvabMK,df.asvabNO,df.asvabPC,df.asvabWK))'
M = fit(PCA, asvabMat; maxoutdim=1)
asvabPCA = MultivariateStats.transform(M, asvabMat)
asvabPCA= vec(asvabPCA')
insert!(df,7, asvabPCA, :asvabPCA)
ols_PCA = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabPCA), df)
println(ols_PCA)

#----------------------------------------------
#5.
asvabMat = (hcat(df.asvabAR, df.asvabCS,df.asvabMK,df.asvabNO,df.asvabPC,df.asvabWK))'
M = fit(FactorAnalysis, asvabMat; maxoutdim=1)
asvabFactorAnalysis = MultivariateStats.transform(M, asvabMat)
asvabFactorAnalysis= vec(asvabFactorAnalysis')
insert!(df,8, asvabFactorAnalysis, :asvabFactorAnalysis)
ols_FactorAnalysis = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + asvabFactorAnalysis), df)
println(ols_FactorAnalysis)

#----------------------------------------------
#6.
function lgwt(N0::Integer,a::Real=-1,b::Real=1) 
    N  = N0-1
    N1 = N+1
    N2 = N+2 
    xu = range(-1,stop=1,length=N1) 
    y = cos.((2*(0:N) .+ 1)*pi/(2*N .+ 2))  .+  ( 0.27/N1 ) .* sin.( pi .* xu .* N/N2 ) 
    L  = zeros(N1,N2) 
    Lp = zeros(N1,N2)  
    y0 = 2    
    vareps = 2e-52
    i = 0
    tracker=0
    it_max=10

    while (norm(y.-y0,Inf)>vareps && tracker<=it_max)
        d=norm(y.-y0,Inf)
    
        L[:,1]  .= 1
        Lp[:,1] .= 0
        
        L[:,2] .= y
        
        for k=2:N1
            L[:,k+1] = ( (2*k-1)*y .* L[:,k] .- (k-1)*L[:,k-1] )/k
        end
        
        Lp = (N2)*( L[:,N1] .- y .* L[:,N2] )./(1 .- y.^2)
        y0 = y
        y  = y0 - L[:,N2]./Lp
        if norm(y.-y0,Inf)==d
            tracker+=1
        end
        i+=1

    end
    
    x = (a.*(1 .- y) .+ b .* (1 .+ y))./2
    w=(b-a)./((1 .- y.^2).*Lp.^2)*(N2/N1)^2
    
    return x,w
end

D= Normal(0,1)
ξ= rand(Normal(0,1),size(df,1),1)
M= hcat(df.asvabAR, df.asvabCS,df.asvabMK,df.asvabNO,df.asvabPC,df.asvabWK)
Xm= hcat(ones(size(df,1)), df.black, df.hispanic, df.female)
X= hcat(ones(size(df,1)), df.black, df.hispanic, df.female, df.schoolt, df.gradHS, df.grad4yr)
Y= df.logwage
L= zeros(size(df,1),1)
function likelihoodfunction(sigma,alpha,beta,gamma,delta)
    sigmaj=sigma[1:end-1]
    sigmaw=sigma[end]
    for i= 1: size(df,1)
        for j= 1:6
            part1vec=zeros(6,1)
            part1vec[j,1]=(1/sigma[1,j])*pdf.(D, (M[i,j]-Xm[i,:]*alpha-gamma*ξ[i,1])/1/sigma[1,j])) 
            part1=part1vec[1,1]*part1vec[2,1]*part1vec[3,1]*part1vec[4,1]*part1vec[5,1]*part1vec[6,1]
        end
        part2=(1/sigmaw)*pdf.(D,(Y[i,1]-X[i,:]*beta-delta*ξ[i,1])/sigmaw)
        L[i,1]= part1*part2
    end
    return L
end

nodes, weights = lgwt(7,-4,4)
likelihood=zeros(size(df,1),1)
for i= 1: size(df,1)
likelihood[i,1]=log(sum(weights.* L[i,1].*pdf.(D,nodes)))
end

sigma_hat = optimize(sigma -> -sum(likelihood), rand(1,7), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(sigma_hat.minimizer)

end

allwrap()