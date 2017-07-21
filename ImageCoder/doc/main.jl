#!/usr/bin/env julia
include("../coder.jl")
using JLD
function main(filename,dst,num_of_centers,kernel)
    Kernels=Dict("gaussian"=> gaussian, "linear"=>linear, 
                 "approximation_l1"=>approximation_l1,
                 "geman_mcclure" => geman_mcclure,"l1ml2"=>l1_l2)
    vbows,A,Y=load_csv(filename)
    #centers,dists,partitions=fft(vbows,Y,num_of_centers)
    #Xm,Xc=center_points(vbows,partitions,centers)
    #println("-> ",length(Xr))
    #println("=> ",length(Xr[1]))
    #JLD.save("codebook.$num_of_centers.jld","centroids",Xm,"centers",Xc )
    data=JLD.load("codebook.sqrt.jld")
    Xm,Xc=data["centroids"],data["centers"]
    Xrm=gen_features(vbows,Xm,Kernels[kernel])
    Xrc=gen_features(vbows,Xc,Kernels[kernel])
    savedb(Xrc,A,"data/centros.sqrt.$kernel.$dst")
    savedb(Xrm,A,"data/centroides.sqrt.$kernel.$dst")
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

    - dst: file name to store new objects or  feature set
      default:ouput.json"""
         )
else
    for filename in ARGS
        main(
            filename,
	    get(ENV, "dst", "output.csv"),
            get(ENV, "num_of_centers", "sqrt"),
            get(ENV, "kernel", "linear"),
        )
    end
end
