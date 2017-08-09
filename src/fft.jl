using SimilaritySearch
using TextModel
using JSON
using DataStructures

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

        if length(L) == 0
            # just link randomly for orthogonal vectors
            n = length(index.db)
            return rand(1:n, floor(Int, log2(n+1)))
        end

        # @show L, fastlinks
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

function create_index(config, distance_function=angle_distance)
    recall=parse(Float64, get(ENV, "recall", "0.90"))
    index = LocalSearchIndex(VBOW, distance_function, recall=recall, neighborhood=LogNeighborhood(1.2))
    index.options.verbose = parse(Bool, get(ENV, "verbose_index", "false"))
    fastlinks = set_oracle(index, Dict{UInt64,KnnResult}())
    return index, fastlinks
end

function full_index(vbows, ind, config, num_fast_links)
    index, fastlinks=create_index(config, cosine)
    for i in ind
        vbow=vbows[i]
        knn, N = find_neighborhood(index, vbow)
        push_neighborhood!(index, vbow, N, length(index.db))
        update_oracle(index, fastlinks, vbow, num_fast_links)
    end

    return index, fastlinks
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


function centroid(vbows::Array{VBOW,1},indices::Array{Int64,1})
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
    return furthest[1][2],furthest[1][1]
end

function maxmin(vbows,centers,ind,index::KnnResult)
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
    index.k=nindex.k
    index.pool=nindex.pool
    fn=pop!(index)
    println(">>>>>>>>>>>>>>> ",fn.objID," ",fn.dist," ",length(index))
    return fn.objID,fn.dist
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
        fid,d=dist[i][2],dist[i][1]
    end
    return fid,d
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
    return fnid,0
end

function  apply_selection(vbows, centers, partitions,ind,selection,index,findex,fastlinks,num_of_fastlinks=7)
    selection_functions=Dict("partition"=>partition,
                             "centroid"=>fcentroid,
                             "maxmin"=>maxmin,
                             "random"=>srandom,
                             "approx_maxmin"=>approx_maxmin)
    if selection == "maxmin"
        fnid,d=selection_functions[selection](vbows,centers,ind,index)
    elseif selection=="approx_maxmin"
        fnid,d=selection_functions[selection](vbows,centers,ind,index,findex)
        vbow=vbows[fnid]
        knn, N = find_neighborhood(index,vbow)
        push_neighborhood!(index, vbow, N, length(index.db))
        update_oracle(index, fastlinks, vbow, length(centers) > 7 ? 7 : length(centers))
    else
        fnid,d=selection_functions[selection](vbows, centers, partitions,ind)
    end
    return fnid,d
end

function approx_maxmin(vbows,centers,ind,index,findex)
    res=PriorityQueue(Int64,Float64)
    #println(">>>>>> Length: ", length(index.db))
    #println(centers)
    if length(centers)==1
        kfn = search(findex, vbows[centers[1]])
        res[first(kfn).objID]=-first(kfn).dist
    else
        for c in centers
        # c=last(centers)
            kfn = search(findex, vbows[c])
            # println("================",Nf,"  ",length(kfn))
            for fn in kfn
                if !(ind[fn.objID] in centers)
                    knn = search(index, vbows[ind[fn.objID]])
                    for nn in knn
                        k, dist = nn.objID, nn.dist
                        if k in keys(res)
                            res[k] = abs(res[k]) < dist ? res[k] : -dist
                        else
                            res[k] = -dist
                        end
                    end
                end
            end
        end
    end
    fnid,d=peek(res)
    while ind[fnid] in centers
        dequeue!(res)
        fnid,d=peek(res)
    end
    return ind[fnid], d
end

function fft(vbows,tweets,config,key,num_of_centers,selection,num_of_fastlinks=7)
    clases=[tweet["klass"] for tweet in tweets]
    indices=[[i for (i,l) in enumerate(clases) if l==c] for c in Set{String}(clases)]
    partitions=[0 for tweet in tweets]
    gcenters=[]
    for ind in indices
        noc=ncenters(num_of_centers,ind)
        centers=[]
        s=rand(1:length(ind))
        push!(centers,ind[s])
        if selection=="approx_maxmin"
            vbow=vbows[ind[s]]
            findex,ffaslinks=full_index(vbows,ind, config, num_of_fastlinks)
            index, fastlinks=create_index(config, angle_distance)
            knn, N = find_neighborhood(index,vbow)
            push_neighborhood!(index, vbow, N, length(index.db))
            update_oracle(index, fastlinks, vbow, 1)
        else
            findex,fastlinks=[],[]
            index=KnnResult(length(ind))
        end
        for k in 1:noc
            fnid,d= apply_selection(vbows, centers, partitions,ind,selection,index,findex,fastlinks)
            push!(centers,fnid)
        end
        gcenters=vcat(gcenters,centers)
    end
    return gcenters
end




# function main(filename,dst,key,num_of_centers,selection)
#     config=TextConfig(); config.nlist=[2]; config.qlist=[5];config.skiplist = []
#     tweets,vbows=list_of_items(filename,key,config)
#     gcenters,partitions=exact_fft(vbows,tweets,config,key,num_of_centers,selection)
#     DBR=toDict(tweets,gcenters,partitions,key)
#     savedb(DBR,dst)
# end

# if length(ARGS) == 0
#     info("""
#     fttc.jl clusters elements using furthest firts travel algorithm

#     Usage: [environment-variables] julia fftc.jl file...

#     Input file must contain a json-dictionary per line

#     For simplicity, the arguments are passed as environment-variables

#     - key: the keyword containing the text for each json
#        default: text

#     - num_of_centers: the number of furthest neighbor used as centers (per class).
#        could be any of log, sqrt or an interger n (0<=n, 0 is equivalent to do rocchio).
#        default: log to indicated that look for log(N) centers, where N is the number
#                 of samples in the input file
#     - dst: file name to store founded  centers
#       default:ouput.json
#     - selection: Criterio used to select the fursthes neigboor, could be any of
#         by partition furthest neighbor (partition) , furthest from centroid(centroid) or maximal min furthest
#         from the already selected heads (maxmin). The two first optios are per partiorion; the
#         last one is global"""

#          )
# else
#     for filename in ARGS
#         main(
#             filename,
# 	    get(ENV, "dst", "output.json"),
#             get(ENV, "key", "text"),
#             get(ENV, "num_of_centers", "log"),
#             get(ENV, "selection", "partition")
#         )
#     end
# end
