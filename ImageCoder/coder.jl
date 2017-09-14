using SimilaritySearch
using TextModel
using JSON
using DataStructures

function savedb(Xr,dst,A=[])
    f=open(dst, "w")
    for i in 1:length(Xr)
        xr=Xr[i]
        if length(A)==0
            data=xr
        else
            data=vcat(xr,A[i])
        end
        write(f,join(data,","))
        write(f,"\n")
    end
    #println(xo[N+1:length(xo)])
end

function suma{T <: Real}(a::Vector{T}, b::Vector{T})::Float64
    sum::T = zero(T)
    @fastmath @inbounds @simd for i in eachindex(a)
        sum += a[i] + b[i]
    end
    return sum
end

function dot{T <: Real}(a::Vector{T}, b::Vector{T})::Float64
    sum::T = zero(T)
    @fastmath @inbounds @simd for i in eachindex(a)
        sum += a[i] * b[i]
    end
    return sum
end


function hist{T <: Real}(a::Vector{T}, b::Vector{T})::Float64
    sum::T = zero(T)
    @fastmath @inbounds @simd for i in eachindex(a)
        sum += minimum(a[i], b[i])
    end
    return sum
end


function load_csv(filename,labels; N=0)
    vbows=[]
    A=[]
    X=readdlm(filename,',')
    n,m=size(X)
    Y=readdlm(labels)
    #println(length(X[1,:]))
    N= N==0 ? m : N
    #println(">>>>>>>>>>>>",N)
    for i in 1:n
        x=X[i,1:N]
        if N!=m
            push!(A,X[i,N+1:m])
        end
        push!(vbows,x)
    end
    return vbows,A,Y
end



function maxmin(vbows,centers,ind,index::KnnResult,distance_function,partitions)
    c=last(centers)
    if length(index)==0
        for i in ind
            if i!=c
                push!(index,i,Inf)
            end
        end
    end
    nindex=KnnResult(length(index))
    for fn in index
        #if !(fn.objID in centers)
        dist=distance_function(vbows[fn.objID],vbows[c])
        dist = if (dist<fn.dist) dist else fn.dist end
        partitions[fn.objID] = if (dist<fn.dist) c else partitions[fn.objID] end
        if fn.objID!=c
            push!(nindex,fn.objID,dist)
        end
    end
    index.k=nindex.k
    index.pool=nindex.pool
    #println(">>>>  ", index)
    fn=pop!(index)
    return fn.objID,fn.dist
end


function ncenters(num_of_centers,ind)
    if num_of_centers=="log"
        noc=trunc(log(2,length(ind)))
    elseif num_of_centers=="sqrt"
        noc=trunc(sqrt(length(ind)))
    else
        noc=parse(Int, num_of_centers)
    end
    return noc
end



function cosine(x1,x2)
    xc1=DenseCosine(x1)
    xc2=DenseCosine(x2)
    d=CosineDistance()
    return d(xc1,xc2)
end

function mcosine(x1,x2,ind;w=0.95)
    n=length(x1)
    k=n-ind
    x=cosine(x1[1:k],x2[1:k])
    y=cosine(x1[k+1:n],x2[k+1:n])
    sim = w*x+(1-w)*y
end


function wL2(x1,x2;ind=4096,w=0.70)
    d=L2Distance()
    n=length(x1)
    k=n-ind
    x=d(x1[1:k],x2[1:k])
    y=d(x1[k+1:n],x2[k+1:n])
    sim = w*x+(1-w)*y
end

function wL2Squared(x1,x2;ind=4096,w=0.70)
    d=L2SquaredDistance()
    n=length(x1)
    k=n-ind
    x=d(x1[1:k],x2[1:k])
    y=d(x1[k+1:n],x2[k+1:n])
    sim = w*x+(1-w)*y
end


function fft(vbows,clases,num_of_centers; distance_function=L2Distance())
    #d=distance_function()
    #clases=[1 for i in 1:length(vbows)]
    indices=[[i for (i,l) in enumerate(clases) if l==c] for c in Set(clases)]
    partitions=[0 for i in 1:length(vbows)]
    gcenters=[]
    dists=[]
    sigmas=Dict()
    cl=[]
    for ind in indices
        noc=ncenters(num_of_centers,ind)-1
        centers=[]
        #s=fchoice([vbows[i] for i in ind],distance_function)
        s=rand(1:length(ind))
        push!(centers,ind[s])
        ll=clases[ind[s]]
        index=KnnResult(length(ind))
        partitions[ind[s]]=ind[s]
        k=1
        while  k<=noc && k<=length(ind)
            fnid,d=maxmin(vbows,centers,ind,index,distance_function,partitions)
            push!(centers,fnid)
            push!(dists,d)
            partitions[fnid]=fnid
            k+=1
        end
        sigmas[ll]=d
        cl=vcat(cl,[ll for i in ind])
        gcenters=vcat(gcenters,centers)
    end
    return gcenters, dists, partitions,sigmas,cl
end

function center_points(vbows, partitions, centers)
    centroids=[]
    for c in centers
        ind=[i for (i,v) in enumerate(partitions) if v==c]
        push!(centroids,mean(vbows[ind]))
    end
    kcenters=vbows[centers]
    return centroids, kcenters
end

function fchoice(X,distance_function)
    c=X[1]
    n=length(X)
    for i in 2:n
        c=X[i]+c
    end
    dist=sort!([(distance_function(x,c),i) for (i,x) in enumerate(X)])
    return first(dist)[2]
end

function linear(xo,xm;sigma=1)
    d=L2Distance()
    return d(xo,xm)
end

function gaussian(xo,xm; distance=L2SquaredDistance(), sigma=1)
    #d=L2SquaredDistance()
    sim=exp(-distance(xo,xm)/(2*sigma))
    sim = isnan(sim) ? 0 : sim
    return sim
end
function cuchy(xo,xm; distance=L2SquaredDistance(), sigma=1)
    #d=L2SquaredDistance()
    sim=1/(1+distance(xo,xm)/sigma*sigma)
    return sim
end

function chi_squared(xo,xm; distance=L2SquaredDistance(), sigma=1)
    #d=L2SquaredDistance()
    sim = 2*dot(xo,xm)/(suma(xo,xm))
    return sim
end

function histogram_intersection(xo,xm; distance=L2SquaredDistance(), sigma=1)
    #d=L2SquaredDistance()
    sim = hist(xo,xm)
    return sim
end

function approximation_l1(xo,xm; sigma=0.01)
    d=L2SquaredDistance()
    return sqrt(d(xo,xm)+sigma*sigma)
end

function geman_mcclure(xo,xm; sigma=1)
    d=L2SquaredDistance()
    dd=d(xo,xm)
    return dd/(dd+sigma*sigma)
end

function l1_l2(xo,xm; sigma=1)
    d=L2SquaredDistance()
    dd=d(xo,xm)
    return 2*(sqrt(1+dd/2)-1)
end

function gen_features(Xo,Xm,sigmas,l; kernel=linear)
    Xr=[]
    nc=length(Xo)
    #println(sigmas)
    dx=L2SquaredDistance()
    dx2=L2Distance()
    for xi in Xo
        xd=[]
        for (j,x) in enumerate(Xm)
            push!(xd, kernel(xi,x,sigma=sigmas[l[j]]))
        end
        push!(Xr,xd)
        #push!(Xr, [kernel(xi,xj) for xj in Xm])
    end
    return Xr
end

