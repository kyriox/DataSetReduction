#!/usr/bin/env julia
include("/home/job/DataSetReduction/ImageCoder/coder.jl")
using JLD
function main(filename,labels,dst,num_of_centers,kernel,codebook,distance_function,weight,split_index)
    N=parse(Int64,split_index)
    if distance_function=="WCosine"
        vbows,A,Y=load_csv(filename,labels)
        Y=[1 for v in vbows]
    else
        vbows,A,Y=load_csv(filename,labels,N=N)
        Y=[1 for v in vbows]
    end
    #X1,Y1=[x for (i,x) in enumerate(vbows) if Y[i]==1],[y for (i,y) in enumerate(Y) if Y[i]==1]
    X1,Y1=vbows,Y
    WCosine(x,y)=mcosine(x,y,N,w=parse(Float64,weight))
    dfunctions=Dict( "L2"=>L2Distance(), "L2Squared"=>L2SquaredDistance(),
                     "WCosine" => WCosine, "cosine"=>cosine)
    df=dfunctions[distance_function]
    cgaussian(x,y;sigma=1)=gaussian(x,y,distance=df,sigma=sigma)
    Kernels=Dict("cgaussian"=> cgaussian,"linear"=>linear, "geman_mcclure" => geman_mcclure,
                   "cuchy"=>cuchy, "hi"=>histogram_intersection, "chi2"=>chi_squared,
                    "gaussian"=> cgaussian)
    kf=Kernels[kernel]
    ws=string(split_index,"N",weight)
    suffix="$num_of_centers.$dst.$kernel.$distance_function.$ws"
    println(suffix)
    if codebook==nothing
        #centers,dists,partitions=fft(vbows,Y,num_of_centers, distance_function=df)
        centers,dists,partitions=fft(X1,Y1,num_of_centers, distance_function=df)
        Xm,Xc=center_points(vbows,partitions,centers)
        JLD.save("codebook.$suffix.jld",
                 "centroids",Xm,"centers",Xc,"dists",dists)
    else
         data=JLD.load(codebook)
         Xm,Xc,dists=data["centroids"],data["centers"],data["dists"]
    end
    #nc=Int64(length(Xm)/2)
    dd=last(dists)
    sigmas=Dict(0=>dd*dd, 1=>dd*dd)
    #l=hcat([0 for i in 1:nc],[1 for i in 1:nc])
    l=Y
    #Xrm=gen_features(vbows,Xm[nc+1:2*nc],sigmas,l,kernel=kf)
    #Xrc=gen_features(vbows,Xc[nc+1:2*nc],sigmas,l,kernel=kf)
    Xrm=gen_features(vbows,Xm,sigmas,l,kernel=kf)
    Xrc=gen_features(vbows,Xc,sigmas,l,kernel=kf)
    println(length(Xrm),"   ", length(Xrc), " ", length(Xrm[1]))
    savedb(Xrc,"centros.$suffix.csv",A)
    savedb(Xrm,"centroides.$suffix.csv",A)
end


if length(ARGS) == 0
    info("""
    coder.jl peform feacture extration and datasets reduction  using   furthest firts travel algorithm

    Usage: [environment-variables] julia fftc.jl file...

    Input file must contain a list of  features per line 

    For simplicity, the arguments are passed as environment-variables

    - num_of_centers: the number of furthest neighbor used as centers (per class).
       could be any of log, sqrt or an interger n (0<=n, 0 is equivalent to do rocchio).
       default: log to indicated that look for log(N) centers, where N is the number
                of samples in the input file

    - kernel: could be any of linear, gaussian, geman_mccluren.
      default: gaussian 
   
    - codebook: path to a yld file with coder values (Centroid/Furthes points list
      and distances vector. If this is not given the codebook is generate and store as 
      codebook.num_of_centers.yld

    - dst: file tail name to store new objects or feature set
      default:ouput.json (this produce two files centros.num_of_centers.kernel.output.cvs
    and centroides.num_of_centers.kernel.output.cvs)
    - distance_function: Cosine, L2 or L2Squared, or WCosine (weigthed cosine).
      default WCosine
    - weight: the weight assigned to the first part of the features vector, the rest is weighted
      1-weigth. This is only required when using wieghted distance function. 
      default 0.95
    - split_index: when using wieghted metric, index specifies where to split the feature vector,
     by know the vector only can split in two slides. 
      default: 0
"""
         )
else
    for filename in ARGS
        main(
            filename,
            get(ENV, "labels", "local_features/labelsz.csv"),
	    get(ENV, "dst", "output.csv"),
            get(ENV, "num_of_centers", "sqrt"),
            get(ENV, "kernel", "gaussian"),
            get(ENV, "codebook", nothing ),
            get(ENV, "distance_function", "L2"),
            get(ENV, "weight", "0"),
            get(ENV, "split_index", "0"),
        )
    end
end
