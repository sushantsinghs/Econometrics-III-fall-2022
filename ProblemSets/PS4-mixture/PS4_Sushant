#Problem Set 4 - Sushant Singh
using Optim, HTTP, GLM, LinearAlgebra, Random, Statistics, DataFrames, CSV, FreqTables
function q6()
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
startvals = [2*rand(7*size(X,2)).-1; .1]
truevals = [0.040374454953787886, 0.24399409314825987, -1.5713213321281576,
            0.04332547647972951, 0.14685584442757782, -2.959104052831784, 
            0.10205742855969692, 0.7473089603290985, -4.120051101278431, 
            0.037562869245200775, 0.6884901936101743, -3.6557712276056433,
            0.020454321679879015, -0.3584005467776664, -4.376931619035733, 
            0.10746368376194072, -0.5263737776237152, -6.199194024108684, 
            0.11688248621330447, -0.28705521539111295, -5.322249141580821, 1.307479115935751]
#  startvals = truevals.*rand(size(truevals))
td = TwiceDifferentiable(theta -> mlogit_with_Z(theta, X, Z, y), startvals; autodiff = :forward)
theta_hat_optim_ad = optimize(td, truevals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
theta_hat_mle_ad = theta_hat_optim_ad.minimizer
H  = Optim.hessian!(td, theta_hat_mle_ad)
theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println(theta_hat_mle_ad) 
println(theta_hat_mle_ad_se) 
println([theta_hat_mle_ad theta_hat_mle_ad_se]) 

#----------------------------------------------
#2.
println("It does make more sense. Estimated γ is 1.307479115935751.")

#----------------------------------------------
#3.
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
    while norm(y.-y0,Inf)>vareps       
        L[:,1]  .= 1
        Lp[:,1] .= 0      
        L[:,2] .= y       
        for k=2:N1
            L[:,k+1] = ( (2*k-1)*y .* L[:,k] .- (k-1)*L[:,k-1] )/k
        end      
        Lp = (N2)*( L[:,N1] .- y .* L[:,N2] )./(1 .- y.^2)
        y0 = y
        y  = y0 - L[:,N2]./Lp
        i+=1
    end
    
    x = (a.*(1 .- y) .+ b .* (1 .+ y))./2   
    w=(b-a)./((1 .- y.^2).*Lp.^2)*(N2/N1)^2   
    return x,w
end

#3(a)
d = Normal(0,1)
nodes, weights = lgwt(7,-4,4)
println(sum(weights.*pdf.(d,nodes)))
println(sum(weights.*nodes.*pdf.(d,nodes)))

#3(b)
d = Normal(0,2) 
nodes, weights = lgwt(7,-5*sqrt(2),5*sqrt(2))
println(sum(weights.*nodes.*nodes.*pdf.(d,nodes)))
nodes, weights = lgwt(10,-5*sqrt(2),5*sqrt(2))
println(sum(weights.*nodes.*nodes.*pdf.(d,nodes)))

#The estimated values are 3.925601931584661 and 3.9769753072954845
#The true value σ^2=4
#The quadrature approximates the true value very well.

#3(c)
D=1000000
μ=0
σ=sqrt(2)
d = Normal(μ,σ^2)
x=rand(Uniform(-5*σ,5*σ),D)
println(sum(((5*σ-(-5*σ))/D).*x.*x.*pdf.(d,x)))
println(sum(((5*σ-(-5*σ))/D).*x.*pdf.(d,x)))
println(sum(((5*σ-(-5*σ))/D).*pdf.(d,x)))

D=1000
x=rand(Uniform(-5*σ,5*σ),D)
println(sum(((5*σ-(-5*σ))/D).*x.*x.*pdf.(d,x)))
println(sum(((5*σ-(-5*σ))/D).*x.*pdf.(d,x)))
println(sum(((5*σ-(-5*σ))/D).*pdf.(d,x)))

#----------------------------------------------
#4.
mu = theta_hat_mle_ad[end]
sigma = theta_hat_mle_ad_se[end]
function mixed_mlogit_with_Z(theta, X, Z, y)
        
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)

    d = Normal(mu,sigma^2)
    nodes, weights = lgwt(7,-4*sigma,4*sigma)
    mixed_p=sum(weights.*P.*pdf.(d,nodes))
    mixed_loglike = -sum( bigY.*log.(mixed_p) )
    
    return mixed_loglike
end

startvals = [2*rand(7*size(X,2)).-1; .1]
td = TwiceDifferentiable(theta ->mixed_loglike, startvals; autodiff = :forward)
mixed_theta_hat_optim_ad = optimize(dt,startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
mixed_theta_hat_mle_ad = mixed_theta_hat_optim_ad.minimizer
H  = Optim.hessian!(td, mixed_theta_hat_mle_ad)
mixed_theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println(mixed_theta_hat_mle_ad) 
println([mixed_theta_hat_mle_ad mixed_theta_hat_mle_ad_se]) 

#----------------------------------------------
#5.
mu = theta_hat_mle_ad[end]
sigma = theta_hat_mle_ad_se[end]
function mc_mixed_mlogit_with_Z(theta, X, Z, y)
        
    alpha = theta[1:end-1]
    gamma = theta[end]
    K = size(X,2)
    J = length(unique(y))
    N = length(y)
    bigY = zeros(N,J)
    for j=1:J
        bigY[:,j] = y.==j
    end
    bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
    
    T = promote_type(eltype(X),eltype(theta))
    num   = zeros(T,N,J)
    dem   = zeros(T,N)
    for j=1:J
        num[:,j] = exp.(X*bigAlpha[:,j] .+ (Z[:,j] .- Z[:,J])*gamma)
        dem .+= num[:,j]
    end
    
    P = num./repeat(dem,1,J)

    d = Normal(mu,sigma^2)
    weights = (4*sigma-(-4*sigma))/size(P)
    mc_mixed_p=sum(weights.*P.*pdf.(d,P))
    mc_mixed_loglike = -sum( bigY.*log.(mc_mixed_p) )
    
    return mc_mixed_loglike
end

startvals = [2*rand(7*size(X,2)).-1; .1]
td = TwiceDifferentiable(theta ->mc_mixed_loglike, startvals; autodiff = :forward)
mc_mixed_theta_hat_optim_ad = optimize(dt,startvals, LBFGS(), Optim.Options(g_tol = 1e-5, iterations=100_000, show_trace=true, show_every=50))
mc_mixed_theta_hat_mle_ad = mc_mixed_theta_hat_optim_ad.minimizer
H  = Optim.hessian!(td, mc_mixed_theta_hat_mle_ad)
mc_mixed_theta_hat_mle_ad_se = sqrt.(diag(inv(H)))
println(mc_mixed_theta_hat_mle_ad) 
println([mc_mixed_theta_hat_mle_ad mc_mixed_theta_hat_mle_ad_se]) 

end
#----------------------------------------------
#6.
q6()