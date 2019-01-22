# function to run a Julia script outside of the current environment
function script(code; wrapper=``, args=``)
    if Base.JLOptions().project != C_NULL
        args = `$args --project=$(unsafe_string(Base.JLOptions().project))`
    end
    mktemp() do path, io
        write(io, code)
        flush(io)
        cmd = `$wrapper $(Base.julia_cmd()) $args $path`
        # redirect stderr to stdout to have it picked up by Weave.jl
        run(pipeline(ignorestatus(cmd), stderr=stdout))
    end
    nothing
end


#' Let's setup a named tuple to hold our model parameters.
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
