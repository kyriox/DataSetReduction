#!/usr/bin/env julia
using SimilaritySearch
using TextModel
using JSON

function set_oracle(index, fastlinks::Dict{UInt64,KnnResult})
    function oracle(q::VBOW)::Vector{Int32}
        L = Int32[]
        for term in q.tokens
            if haskey(fastlinks, term.id)
                for p in fastlinks[term.id]
                    push!(L, p.objID)
                end
            end
        end
        L
    end
    index.options.oracle = oracle
    fastlinks
end

function update_oracle(index, fastlinks, bow::VBOW, num_fast_links)
    for term in bow.tokens
        if !haskey(fastlinks, term.id)
            fastlinks[term.id] = KnnResult(num_fast_links)
        end
        push!(fastlinks[term.id], length(index.db), term.weight)
    end
end

function create_index(config)
    index = LocalSearchIndex(VBOW, angle_distance, recall=0.90, neighborhood=LogSatNeighborhood(1.5))
    fastlinks = set_oracle(index, Dict{UInt64,KnnResult}())
    return index, config, fastlinks
end

function savedb(DBR, dst)
    f=open(dst, "w")
    for twk in keys(DBR)
        #printlnt(twk)
    	JSON.print(f,DBR[twk])
	write(f,"\n")
    end
end

function list_of_items(filename,key,config) 
    tweets=[]
    bows=[]
    vbows=[]
    mapa=Dict("positive"=>"P", "negative"=>"N", "neutral"=>"NEU",
              "P"=>"P", "N"=>"N", "NEU"=>"NEU", "NONE"=>"NONE")
    iterlines(filename) do line
        tw=TextModel.parsetweet(line)
        tw["klass"]=mapa[tw["klass"]]
        bow=compute_bow(tw[key], config) |> VBOW
        vbow=VBOW(bow)
        if length(vbow)>0
            push!(tweets,tw)
            push!(bows,bow)
            push!(vbows,VBOW(bow))
        end
    end
    return tweets,vbows
end

function assign(vbows,partitions,centers,ind)
    if length(centers)==1
        for i in ind
            partitions[i]=centers[1]
        end
    else
        for i in ind
            dist=sort([(angle_distance(vbows[c],vbows[i]),c) for c in centers])
            partitions[i]=dist[1][2]
        end        
    end
end


function centroid(vbows,indices)
    N=length(indices)
    t=vbows[indices[1]]
    for i in 2:N
        t=t+vbows[i]
    end
    return t*(1/N)
end

function get_indices(partitions,ind,k)
    indices=Dict()
    for i in ind
        if haskey(indices,i)
            push!(indices[i], )
        end
        push!(indices, [c for c in ind if partitions[c]==idc])
    end
    return indices
end

function partition(vbows,centers,partitions,ind)
    furthest=[]
    assign(vbows,partitions,centers,ind)
    for c in centers
        cind=[i for (i,l) in enumerate(partitions) if (l==c && i!=c)]
        dist=sort([(cosine(vbows[i],vbows[c]),i) for i in cind])
        if length(dist)>0
            push!(furthest, dist[1])
        end
    end
    sort!(furthest)
    return furthest[1][2]
end

function maxmin(vbows,centers,ind,index)
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
        dist=angle_distance(vbows[fn.objID],vbows[c])
        dist = if (dist<fn.dist) dist else fn.dist end 
        if fn.objID!=c
            push!(nindex,fn.objID,dist)
        end
    end
    index=nindex
    fn=pop!(index)
    return index,fn.objID
end

function fcentroid(vbows,centers,partitions,ind)
    total=vbows[centers[1]]
    for c in centers[2:length(centers)]
        total=total+vbows[c]
    end
    dist=sort([(cosine(vbows[i], total),i) for i in ind if !(i in centers)])
    i=1
    fid=dist[1][2]
    while fid in centers
        i=i+1
        fid=dist[i][2]
    end 
    return fid
end

function toDict(tweets, centers, partitions, key)
    DBR=Dict()
    #Select head element
    for k in centers
        #k=findfirst(partitions, i)
        twk=copy(tweets[k])
        #twk[key]=[]
        DBR[k]=twk
    end
    #Group elements in the same group 
    # for (i,v) in enumerate(partitions)
    #     k=findfirst(partitions, v)
    #     push!(DBR[k][key],tweets[i][key])
    # end
    return DBR
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

function srandom(vbows, centers, partitions,ind)
    s=rand(1:length(ind))
    fnid=ind[s]
    while fnid in centers
        s=rand(1:length(ind))
        fnid=ind[s]
    end
    return fnid
end

function  apply_selection(vbows, centers, partitions,ind,selection,index)
    selection_functions=Dict("partition"=>partition,
                             "centroid"=>fcentroid,
                             "maxmin"=>maxmin,
                             "random"=>srandom)
    if selection == "maxmin"
        index,fnid=selection_functions[selection](vbows,centers,ind,index)
    else
        fnid=selection_functions[selection](vbows, centers, partitions,ind)
    end
    return index,fnid
end

function exact_fft(vbows,tweets,config,key,num_of_centers,selection)
    clases=[tweet["klass"] for tweet in tweets]
    indices=[[i for (i,l) in enumerate(clases) if l==c] for c in Set{String}(clases)]
    partitions=[0 for tweet in tweets]
    gcenters=[]
    for ind in indices
        noc=ncenters(num_of_centers,ind)
        centers=[]
        s=rand(1:length(ind))
        push!(centers,ind[s])
        index=KnnResult(length(ind))
        for k in 1:noc
        #f=true
        #med=2
        #while f || (med>1 && length(index)>1)
            index,fnid=apply_selection(vbows, centers, partitions,ind,selection,index)
            push!(centers,fnid)
            if length(centers)%1000==0
                println(last(index).dist," ",fnid)
                println(length(centers)," ",length(index))
                xx=[k.dist for k in index]
                med=median(xx)
                println(mean(xx)," ",med)
            end
        end
       gcenters=vcat(gcenters,centers)
    end
    sc=sort(gcenters)
    #println(sc)
    #assign(vbows,partitions,gcenters,ind)
    println("Termine!")
    return gcenters,partitions
end

function approx_maxmin(vbows,centers,ind,index,findex)
    for i in range(1:length(centers))
        knn, N = find_neighborhood(index, vbows[i])
        j=1
        k=ind[knn[j].objID]
        while (k in centers)
            j=j+1
            k=ind[knn[j].objID]
        end
        push!(furthest,(-1*knn[j].dist, ind[knn[j].objID]))
    end
    return first(furthest)[2]
end

function fcentroid(vbows,centers,partitions,ind,index)
    total=vbows[centers[1]]
    for c in centers[2:length(centers)]
        total=total+vbows[c]
    end
    knn, N = find_neighborhood(index, total)
    j=1
    k=ind[knn[j].objID]
    while (k in centers)
        j=j+1
        k=ind[knn[j].objID]
    end
    push!(furthest,(-1*knn[j].dist, ind[knn[j].objID]))
    return first(furthest)[2]
end

function approx_fft(vbows,tweets,config,key,num_of_centers,selection)
    index=create_
    clases=[tweet["klass"] for tweet in tweets]
    indices=[[i for (i,l) in enumerate(clases) if l==c] for c in Set{String}(clases)]
    partitions=[0 for tweet in tweets]
    gcenters=[]
    for ind in indices
        noc=ncenters(num_of_centers,ind)
        centers=[]
        s=rand(1:length(ind))
        push!(centers,ind[s])
        for k in 1:noc
            fnid=apply_selection(vbows, centers, partitions,ind,selection)
            push!(centers,fnid)
        end
       gcenters=vcat(gcenters,centers)
    end
    sc=sort(gcenters)
    #println(sc)
    assign(vbows,partitions,gcenters,ind)
    return gcenters,partitions
end

function main(filename,dst,key,num_of_centers,selection)
    config=TextConfig(); config.nlist=[2]; config.qlist=[5];config.skiplist = []
    tweets,vbows=list_of_items(filename,key,config)
    gcenters,partitions=exact_fft(vbows,tweets,config,key,num_of_centers,selection)
    DBR=toDict(tweets,gcenters,partitions,key)
    savedb(DBR,dst)
end

if length(ARGS) == 0
    info("""
    fttc.jl clusters elements using furthest firts travel algorithm  

    Usage: [environment-variables] julia fftc.jl file...

    Input file must contain a json-dictionary per line

    For simplicity, the arguments are passed as environment-variables

    - key: the keyword containing the text for each json
       default: text

    - num_of_centers: the number of furthest neighbor used as centers (per class). 
       could be any of log, sqrt or an interger n (0<=n, 0 is equivalent to do rocchio).
       default: log to indicated that look for log(N) centers, where N is the number 
                of samples in the input file
    - dst: file name to store founded  centers
      default:ouput.json
    - selection: Criterio used to select the fursthes neigboor, could be any of
        by partition furthest neighbor (partition) , furthest from centroid(centroid) or maximal min furthest 
        from the already selected heads (maxmin). The two first optios are per partiorion; the 
        last one is global"""
    
         )
else
    for filename in ARGS
        main(
            filename,
	    get(ENV, "dst", "output.json"),	
            get(ENV, "key", "text"),
            get(ENV, "num_of_centers", "log"),
            get(ENV, "selection", "partition")
        )
    end
end
