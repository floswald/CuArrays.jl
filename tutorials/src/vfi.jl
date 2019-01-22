#' ---
#' title : Using the GPU to do Value Function Iteration
#' author : Florian Oswald
#' ---


#' This tutorial is inspired [Tim Holy's introduction to GPU for Julia](https://juliagpu.gitlab.io/CuArrays.jl/tutorials/intro.html). Here we implement a custom Kernel to compute a simple value function iteration (VFI) algorithm. That part of the tutorial is based on [Fernandez-Villaverde and Zarruk's](https://github.com/davidzarruk/Parallel_Computing) parallel computing survey for Economists. For a good introduction to the computational side of value function iteration, I recommend the corresponding [QuantEcon tutorial](https://lectures.quantecon.org/jl/discrete_dp.html).



#+ echo=false; results="hidden"

include(joinpath(@__DIR__, "common.jl"))

#+

#' # Setup

#' We create a function to build our parameter vector.

#+ eval=false"

function param(;nx=1500,ne=15,T=10)
    p = (
        nx         = nx,
        xmax       = 4.0,
        xmin       = 0.1,
        ne         = ne,
        sigma_eps  = 0.02058,
        lambda_eps = 0.99,
        m          = 1.5,
        sigma      = 2,
        beta       = 0.97,
        T          = 10,
        r          = 0.07,
        w          = 5,
        xgrid      = zeros(Float32,nx),
        egrid      = zeros(Float32,ne),
        P          = zeros(Float32,ne,ne))


    # populate transition matrix

    # Grid for capital (x)
    p.xgrid[:] = collect(range(p.xmin,stop=p.xmax,length=p.nx));

    # Grid for productivity (e) with Tauchen (1986)
    sigma_y = sqrt((p.sigma_eps^2) / (1 - (p.lambda_eps^2)));
    estep = 2*sigma_y*p.m / (p.ne-1);
    for i = 1:p.ne
        p.egrid[i] = (-p.m*sqrt((p.sigma_eps^2) / (1 - (p.lambda_eps^2))) + (i-1)*estep);
    end

    # Transition probability matrix (P) Tauchen (1986)
    mm = p.egrid[2] - p.egrid[1];
    for j = 1:p.ne
        for k = 1:p.ne
            if(k == 1)
                p.P[j, k] = cdf(Normal(), (p.egrid[k] - p.lambda_eps*p.egrid[j] + (mm/2))/p.sigma_eps);
            elseif(k == ne)
                p.P[j, k] = 1 - cdf(Normal(), (p.egrid[k] - p.lambda_eps*p.egrid[j] - (mm/2))/p.sigma_eps);
            else
                p.P[j, k] = cdf(Normal(), (p.egrid[k] - p.lambda_eps*p.egrid[j] + (mm/2))/p.sigma_eps) - cdf(Normal(), (p.egrid[k] - p.lambda_eps*p.egrid[j] - (mm/2))/p.sigma_eps);
            end
        end
    end

    # exponential of egrid
    p.egrid[:] = exp.(p.egrid)
    return p
end

#+

#' let's get an instance of it.

using Distributions
p = param()



#' ## Computing the value function at a single state

#' A *state* in our setup is a triple `(ix,ie,age)`, i.e. a value of capital and endowment shock. Here is how we could do this:

function v(ix::Int,ie::Int,age::Int,Vnext::Matrix{Float64},p::NamedTuple)

    out = typemin(eltype(Vnext))
    ixpopt = 0 # optimal policy
    expected = 0.0
    utility = out
    for jx in 1:p.nx
        if age < p.T
            for je in 1:p.ne 
                expected += p.P[ie,je] * Vnext[jx,je]
            end
        end
        cons = (1+p.r)*p.xgrid[ix] + p.egrid[ie]*p.w - p.xgrid[jx]
        utility = (cons^(1-p.sigma))/(1-p.sigma) + p.beta * expected
        if cons <= 0
            utility = typemin(eltype(Vnext)) 
        end
        if utility >= out
            out = utility
            ixpopt = jx
        end
    end
    return out 
end

#' and finally, a function that cycles over age and each state

function cpu_lifecycle(p::NamedTuple)
    V = zeros(p.nx,p.ne,p.T)
    Vnext = zeros(p.nx,p.ne)
    for age in p.T:-1:1
        for ix in 1:p.nx
            for ie in 1:p.ne
                value = v(ix,ie,age,Vnext,p)
                V[ix,ie,age] = value
            end
        end
        Vnext[:,:] = V[:,:,age]
    end
    println("first three shock states at age 1: $(V[1,1:3,1])")
    return V
end

vcpu = cpu_lifecycle(p);

#' Let's make a plot of the resulting value function:

using Plots
CList = reshape( range(colorant"yellow", stop=colorant"red",length=p.ne), 1, p.ne );
p1 = plot(p.xgrid,vcpu[:,:,1],
    linecolor=CList,
    title="period 1",
    leg=false,label=["e$i" for i in 1:p.ne],xlabel="x",ylabel="v(x)")
p2 = plot(p.xgrid,vcpu[:,:,p.T-1],
    linecolor=CList,
    title="period T-1",
    leg=:bottomright,label=["e$i" for i in 1:p.ne],xlabel="x",ylabel="v(x)")
plot(p1,p2,layout=2, link=:y)

  
#' ## Parallelizing on the CPU  

#' ### Multicore 

#' Multicore computation distributes the work across multiple worker nodes. 
#' A slight complication with the distributed model is that the worker processes use
#' separate memory - even if they reside in the same machine! I find it easiest to imagine that you
#' start several separate julia processes in your terminal, each in their own tab. Clearly, 
#' creating `x=2` in the first processes doesn't make the object `x` available in the other processes. 
#' This is true for data (like `x`), as it is for code. Any function you want to call on one of the worker
#' processes must be *defined* on that process. Again visualize the separate terminal windows, 
#' and imagine defining function `f()` on process 1. Calling `f()` on process 2 will return an error. 
#' This is illustrated here:
#'
#' ![processes](iterms.png)
#'
#'
#' ![block grid](intro1.png)
#'
#' The `Distributed` package provides functionality to efficiently connect those separate processes, 
#' and the `SharedArrays` package gives us an `Array` type which may be shared by several processes.
#' The function `workers()` shows the ids of the currently active workers. By default, this is 
#' just a single worker, which we also often call the *master* node:


using SharedArrays
using Distributed

workers()
addprocs(5)  # add 5 workers

#' Let's define a function that distributes the computation of each state (ix,ie) over our set of workers. the main thing to look out for is that we have to make the function `v` available on all workers. We could save it to a file `script.jl`, for example, and load julia with the command line option `--load script.jl`. Alternatively, we redefine it here, by prefix with the macro `@everywhere`:

@everywhere function v(ix::Int,ie::Int,age::Int,Vnext::Matrix{Float64},p::NamedTuple)

    out = typemin(eltype(Vnext))
    ixpopt = 0 # optimal policy
    expected = 0.0
    utility = out
    for jx in 1:p.nx
        if age < p.T
            for je in 1:p.ne 
                expected += p.P[ie,je] * Vnext[jx,je]
            end
        end
        cons = (1+p.r)*p.xgrid[ix] + p.egrid[ie]*p.w - p.xgrid[jx]
        utility = (cons^(1-p.sigma))/(1-p.sigma) + p.beta * expected
        if cons <= 0
            utility = typemin(eltype(Vnext)) 
        end
        if utility >= out
            out = utility
            ixpopt = jx
        end
    end
    return out 
end



#' Now that this is defined on all workers, we can move on to define the calling function on the master node. Notice that we have to use a `SharedArray` for the worker processes to save their results in - remember that by default they would not have access to a standard `Array` created on the master worker.

function cpu_lifecycle_multicore(p::NamedTuple)
    println("julia running with $(length(workers())) workers")
    V = zeros(p.nx,p.ne,p.T)
    Vnext = zeros(p.nx,p.ne)
    Vshared = SharedArray{Float64}(p.nx,p.ne)   # note!
    for age in p.T:-1:1
        @sync @distributed for i in 1:(p.nx*p.ne)   # distribute over a single index
            ix,ie = Tuple(CartesianIndices((p.nx,p.ne))[i])
            value = v(ix,ie,age,Vnext,p)
            Vshared[ix,ie] = value
        end
        V[:,:,age] = Vshared
        Vnext[:,:] = Vshared
    end
    println("first three shock states at age 1: $(V[1,1:3,1])")
    return V
end

#' **Important**: Notice that the worker processes do *not* have to reside on the same machine, 
#' as in this example was indeed the case. The function `addprocs` is very powerful, so check out `?addprocs`. 
#' In particular, one can simply supply a list of machine identifiers (like for instance IP addresses) 
#' to the function. It is interesting to look at the output of `htop` (or Activity Monitor on OSX) 
#' and see several different julia processes doing work, when we call this function:  
#' 
#' ![addprocs](addprocs.png)
#' 


#' ### Multithread 

#' Related to this is the multithread model of parallelization. The good thing here is that we operate in shared memory, i.e. we don't have worry about moving data or code around - there is just a single julia session running in your imaginary terminal from above. What is happening, however, is that julia operates several *threads* through our CPU, whenever we tell it to do so with the macro `Threads.@threads`. One needs to be very careful to write such code in *thread-safe* way, that is, no 2 threads should want to write to the same array index, for example. 
#' Similarly to having to add workers via `addprocs` as above, here we have to start julia with multiple threads. You can achieve that by starting julia via 


#+ eval=false

JULIA_NUM_THREADS=5 julia

#+

#' Here is our lifecycle problem with a threaded loop:

function cpu_lifecycle_thread(p::NamedTuple)
    println("julia running with $(Threads.nthreads()) threads")
    V = zeros(p.nx,p.ne,p.T)
    Vnext = zeros(p.nx,p.ne)
    for age in p.T:-1:1
        Threads.@threads for i in 1:(p.nx*p.ne)
            ix,ie = Tuple(CartesianIndices((p.nx,p.ne))[i])
            value = v(ix,ie,age,Vnext,p)
            V[ix,ie,age] = value
        end
        Vnext[:,:] = V[:,:,age]
    end
    println("first three shock states at age 1: $(V[1,1:3,1])")
    return V
end

#' and, again, it's instructive to look at the system monitor via `htop` or similar. 
#' Now we see that a single process (the process that builds this page, executing `julia make.jl`) 
#' runs at almost 400% CPU utilization. The other worker processes are idle.  
#' 
#' ![threads](threads.png)
#' 


#' ### CPU Benchmarks

#' Before worrying about speed, we always want to verify that our computations are accurate. Taking the content of `vcpu` as the truth, let's see how close we get with the threaded computation:

vthread = cpu_lifecycle_thread(p);

using Test
@test vthread ≈ vcpu


#' You can see that I used the symbol `≈` here, which means *approximately equal*. In the `julia` REPL you obtain this by typing `\approx<TAB>`. Now let's worry about speed. Benchmarking incurs machine noise. This is very similar to any statistical sampling proceedure. Consider those timings obtained from a naive timing, for example:

@time cpu_lifecycle(p);
@time cpu_lifecycle(p);
@time cpu_lifecycle(p);


#' You can see that they all differ. In order to improve on this, we have the `BenchmarkTools` package available. It basically runs the benchmark function many times and reports summary statistics.:

using BenchmarkTools
b1 = @benchmark cpu_lifecycle(p);
b2 = @benchmark cpu_lifecycle_thread(p);
b1
b2

#+ echo=false

run(`$(Base.julia_cmd()) $(joinpath(@__DIR__, "addprocs-vfi.jl"))`)

#+




#' So, we can see that depending


#' ## Parallelizing on the GPU 

# using CuArrays
# using CUDAnative

# function v_gpu!(V,nx,ne,T,xgrid,egrid,r,w,sigma,beta,P,age)

#     # get index 
#     # ix = 1
#     # ie = 4

#     badval = -1_000_000.0f0
#     vtmp = badval  # intiate V at lowest value
#     ixpopt = 0 # optimal policy
#     expected = 0.0f0
#     utility = badval

#     # strided loop
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = blockDim().x * gridDim().x
    
#     for i = index:stride:(nx*ne)

#         ix = ceil(Int,i/ne)
#         ie = ceil(Int,mod(i-0.05,ne))
#         # @cuprintf("threadIdx %ld, blockDim %ld,ix %ld, ie %ld\n", index, stride,ix,ie)
#         j = ix + nx*(ie-1 + ne*(age-1))
#         vtmp = badval  # intiate V at lowest value
#         for jx in 1:nx
#             if age < T
#                 for je in 1:ne 
#                     expected = expected + P[ie,je] * V[jx + nx*(je-1 + ne*(age))]
#                 end
#             end
#             cons = convert(Float32,(1+r)*xgrid[ix] + egrid[ie]*w - xgrid[jx])
#             utility = CUDAnative.pow(cons,convert(Float32,(1-sigma)))/(1-sigma) + beta * expected
#             if cons <= 0
#                 utility = badval
#             end
#             if utility >= vtmp
#                 vtmp = utility
#             end
#         end
#         V[j] = vtmp

#     end
#     return nothing
# end

# V_d = CuArray(zeros(Float32,nx,ne,T));
# xgrid_d = CuArray(p.xgrid);
# egrid_d = CuArray(p.egrid);
# P_d = CuArray(p.P);

# block_size = 256

# @cuda threads=ceil(Int,(nx*ne)/block_size) blocks=block_size v_gpu!(V_d,p.nx,p.ne,p.T,xgrid_d,egrid_d,p.r,p.w,p.sigma,p.beta,P_d,T)

# function bench_v_gpu!(V_d,nx,ne,T,xgrid_d,egrid_d,r,w,sigma,beta,P_d,age,block_size)
#     CuArrays.@sync begin
#         @cuda threads=ceil(Int,(nx*ne)/block_size) blocks=block_size v_gpu!(V_d,nx,ne,T,xgrid_d,egrid_d,r,w,sigma,beta,P_d,age)
#     end
# end

# bench_v_gpu!(V_d,p.nx,p.ne,p.T,xgrid_d,egrid_d,p.r,p.w,p.sigma,p.beta,P_d,2,block_size);
# @btime bench_v_gpu!(V_d,p.nx,p.ne,p.T,xgrid_d,egrid_d,p.r,p.w,p.sigma,p.beta,P_d,2,block_size)

# # using CUDAdrv 
# # CUDAdrv.@profile bench_v_gpu!(V_d,p.nx,p.ne,p.T,xgrid_d,egrid_d,p.r,p.w,p.sigma,p.beta,P_d,2,block_size)
# # exit()


# #' over the lifecycle

# function gpu_lifecycle(p::NamedTuple)
#     V_d = CuArray(zeros(Float32,nx,ne,T));
#     xgrid_d = CuArray(p.xgrid);
#     egrid_d = CuArray(p.egrid);
#     P_d = CuArray(p.P);

#     block_size = 256

#     for it in p.T:-1:1
#         @cuda threads=ceil(Int,(nx*ne)/block_size) blocks=block_size v_gpu!(V_d,p.nx,p.ne,p.T,xgrid_d,egrid_d,p.r,p.w,p.sigma,p.beta,P_d,it)
#     end
#     out = Array(V_d)  # download data from GPU 
#     return out
# end

# @btime gpu_lifecycle(p);
# @btime cpu_lifecycle(p);


# #' how costly is it to create the `CuArray`s? Let's create them beforehand. 

# function gpu_lifecycle2(p::NamedTuple,V_d,xgrid_d,egrid_d,P_d)

#     block_size = 256

#     for it in p.T:-1:1
#         @cuda threads=ceil(Int,(nx*ne)/block_size) blocks=block_size v_gpu!(V_d,p.nx,p.ne,p.T,xgrid_d,egrid_d,p.r,p.w,p.sigma,p.beta,P_d,it)
#     end
#     out = Array(V_d)  # download data from GPU 
#     return out
# end

# V_d = CuArray(zeros(Float32,nx,ne,T));
# xgrid_d = CuArray(p.xgrid);
# egrid_d = CuArray(p.egrid);
# P_d = CuArray(p.P);

# @btime gpu_lifecycle2(p,V_d,xgrid_d,egrid_d,P_d);
# @btime cpu_lifecycle(p);


# #' not overly costly. how about moving the time iteration loop onto the GPU instead?


# function v_gpu_age!(V,nx,ne,T,xgrid,egrid,r,w,sigma,beta,P)

#     # get index 
#     # ix = 1
#     # ie = 4

#     badval = -1_000_000.0f0
#     vtmp = badval  # intiate V at lowest value
#     ixpopt = 0 # optimal policy
#     expected = 0.0f0
#     utility = badval

#     # strided loop
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = blockDim().x * gridDim().x

#     # time loop
#     for age in T:-1:1

#         for i = index:stride:(nx*ne)

#             ix = ceil(Int,i/ne)
#             ie = ceil(Int,mod(i-0.05,ne))
#             # @cuprintf("threadIdx %ld, blockDim %ld,ix %ld, ie %ld\n", index, stride,ix,ie)
#             j = ix + nx*(ie-1 + ne*(age-1))
#             vtmp = badval  # intiate V at lowest value
#             for jx in 1:nx
#                 if age < T
#                     for je in 1:ne 
#                         expected = expected + P[ie,je] * V[jx + nx*(je-1 + ne*(age))]
#                     end
#                 end
#                 cons = convert(Float32,(1+r)*xgrid[ix] + egrid[ie]*w - xgrid[jx])
#                 utility = CUDAnative.pow(cons,convert(Float32,(1-sigma)))/(1-sigma) + beta * expected
#                 if cons <= 0
#                     utility = badval
#                 end
#                 if utility >= vtmp
#                     vtmp = utility
#                 end
#             end
#             V[j] = vtmp
#         end
#     end  # time
    
    
#     return nothing
# end

# function gpu_lifecycle_age(p::NamedTuple)
#     V_d = CuArray(zeros(Float32,nx,ne,T));
#     xgrid_d = CuArray(p.xgrid);
#     egrid_d = CuArray(p.egrid);
#     P_d = CuArray(p.P);

#     block_size = 256

#     @cuda threads=ceil(Int,(nx*ne)/block_size) blocks=block_size v_gpu_age!(V_d,p.nx,p.ne,p.T,xgrid_d,egrid_d,p.r,p.w,p.sigma,p.beta,P_d)
#     out = Array(V_d)  # download data from GPU 
#     return out
# end



# @btime gpu_lifecycle_age(p);
# @btime cpu_lifecycle(p);


# # We could again benchmark this with `nvprof

# using CUDAdrv 
# CUDAdrv.@profile gpu_lifecycle_age(p);


# code = """
#     using CUDAdrv, CUDAnative, CuArrays
#     using Distributions
#     include(joinpath(@__DIR__, "common.jl"))

#     p = param()
    
#     function v_gpu_age!(V,nx,ne,T,xgrid,egrid,r,w,sigma,beta,P)

#         # get index 
#         # ix = 1
#         # ie = 4

#         badval = -1_000_000.0f0
#         vtmp = badval  # intiate V at lowest value
#         ixpopt = 0 # optimal policy
#         expected = 0.0f0
#         utility = badval

#         # strided loop
#         index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#         stride = blockDim().x * gridDim().x

#         # time loop
#         for age in T:-1:1

#             for i = index:stride:(nx*ne)

#                 ix = ceil(Int,i/ne)
#                 ie = ceil(Int,mod(i-0.05,ne))
#                 # @cuprintf("threadIdx %ld, blockDim %ld,ix %ld, ie %ld\n", index, stride,ix,ie)
#                 j = ix + nx*(ie-1 + ne*(age-1))
#                 vtmp = badval  # intiate V at lowest value
#                 for jx in 1:nx
#                     if age < T
#                         for je in 1:ne 
#                             expected = expected + P[ie,je] * V[jx + nx*(je-1 + ne*(age))]
#                         end
#                     end
#                     cons = convert(Float32,(1+r)*xgrid[ix] + egrid[ie]*w - xgrid[jx])
#                     utility = CUDAnative.pow(cons,convert(Float32,(1-sigma)))/(1-sigma) + beta * expected
#                     if cons <= 0
#                         utility = badval
#                     end
#                     if utility >= vtmp
#                         vtmp = utility
#                     end
#                 end
#                 V[j] = vtmp
#             end
#         end  # time
        
        
#         return nothing
#     end

#     function gpu_lifecycle_age(p::NamedTuple)
#         V_d = CuArray(zeros(Float32,nx,ne,T));
#         xgrid_d = CuArray(p.xgrid);
#         egrid_d = CuArray(p.egrid);
#         P_d = CuArray(p.P);

#         block_size = 256

#         @cuda threads=ceil(Int,(nx*ne)/block_size) blocks=block_size v_gpu_age!(V_d,p.nx,p.ne,p.T,xgrid_d,egrid_d,p.r,p.w,p.sigma,p.beta,P_d)
#         out = Array(V_d)  # download data from GPU 
#         return out
#     end

#     gpu_lifecycle_age(p);

#     CUDAdrv.@profile gpu_lifecycle_age(p);
# """

# script(code; wrapper=`nvprof --unified-memory-profiling off --profile-from-start off --print-gpu-trace`)

