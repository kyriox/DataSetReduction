using SimilaritySearch
using TextModel
using JSON
using DataStructures

function savedb(Xr, A, dst, N=4096)
    f=open(dst, "w")
    println(length(Xr)," ",length(A))
    println(length(Xr[1])," ",length(A[1]))
    for i in 1:length(Xr)
        xr,xo=Xr[i],A[i]
        data=vcat(xr,xo)	
        write(f,join(data,","))
        write(f,"\n")
    end
    println(xo[N+1:length(xo)])
end

function load_csv(filename,N=4096)
    vbows=[]
    A=[]
    X=transpose(hcat([[parse(Float64,d) for d in split(x,",")] for x in readdlm(filename)]...))
    n,m=size(X)
    println(n,"  ",m)
    Y=X[:,m]
    for i in 1:n
        x=X[i,1:N]
        push!(A,X[i,N+1:m])
    #    #bow=Dict(zip(["a$j" for j in x],x))
    #     bow=Dict(zip(["a$j" for (j,v) in enumerate(x)],x))
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


function fft(vbows,clases,num_of_centers,distance_function=L2Distance())
    #d=distance_function()
    clases=[1 for i in 1:length(vbows)]
    indices=[[i for (i,l) in enumerate(clases) if l==c] for c in Set(clases)]
    partitions=[0 for vbow in vbows]
    gcenters=[]
    dists=[]
    for ind in indices
        noc=ncenters(num_of_centers,ind)-1
        centers=[]
        s=rand(1:length(ind))
        push!(centers,ind[s])
        index=KnnResult(length(ind))
        partitions[ind[s]]=ind[s]
        for k in 1:noc
            fnid,d=maxmin(vbows,centers,ind,index,distance_function,partitions)
            push!(centers,fnid)
            push!(dists,d)
            partitions[fnid]=fnid
        end
        gcenters=vcat(gcenters,centers)
    end
    return gcenters, dists, partitions
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

function linear(xo,xm)
    d=L2Distance()
    return d(xo,xm)
end

function gaussian(xo,xm,gamma=1)
    d=L2SquaredDistance()
    sim=exp(-d(xo,xm)*gamma)
    return sim
end

function approximation_l1(xo,xm,beta=0.01)
    d=L2SquaredDistance()
    return sqrt(d(xo,xm)+beta*beta)
end

function geman_mcclure(xo,xm)
    d=L2SquaredDistance()
    dd=d(xo,xm)
    return dd/(2+2*dd)
end

function l1_l2(xo,xm)
    d=L2SquaredDistance()
    dd=d(xo,xm)
    return 2*(sqrt(1+dd/2)-1)
end

function gen_features(Xo,Xm,kernel=linear)
    Xr=[]
    for xi in Xo
        push!(Xr, [kernel(xi,xj) for xj in Xm])
    end
    return Xr
end

