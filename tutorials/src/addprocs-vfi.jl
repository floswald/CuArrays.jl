include(joinpath(@__DIR__, "common.jl"))
    using Distributions
    using BenchmarkTools
    using SharedArrays
    using Distributed
    
    p = param()

    addprocs(5)  # add 5 workers
    using Weave
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
        return V
    end

    println("@time cpu_lifecycle_multicore(p)")
    V = cpu_lifecycle_multicore(p);
    println("first three shock states at age 1: $(V[1,1:3,1])")
    @time cpu_lifecycle_multicore(p)    