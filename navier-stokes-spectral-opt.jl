using FFTW
using Plots

using HDF5

"""
An optimized version of the Navier-Stokes spectral method simulation
"""

struct SimulationParams
    kx::Array{Float64, 2}
    ky::Array{Float64, 2}
    kmax::Float64
    kSq::Array{Float64, 2}
    kSq_inv::Array{Float64, 2}
    dealias::Array{Bool, 2}
end

function build_params(N::Int64, L::Float64)
    klin = 2.0 * pi / L * collect(-N/2:N/2-1)
    kmax = maximum(klin)
    kx, ky = repeat(klin', N, 1), repeat(klin, 1, N)
    kx = ifftshift(kx)
    ky = ifftshift(ky)
    kSq = kx.^2 + ky.^2
    kSq_inv = 1.0 ./ kSq
    kSq_inv[kSq .== 0] .= 1
    dealias = (abs.(kx) .< (2.0/3.0) * kmax) .& (abs.(ky) .< (2.0/3.0) * kmax)
    return SimulationParams(kx, ky, kmax, kSq, kSq_inv, dealias)
end

function poisson_solve(rho::Array{Float64, 2}, params::SimulationParams)
    V_hat = -fft(rho) .* params.kSq_inv
    V = real(ifft(V_hat))
    return V
end

function diffusion_solve(v::Array{Float64, 2}, dt::Float64, nu::Float64, params::SimulationParams)
    v_hat = fft(v) ./ (1.0 .+ dt * nu .* params.kSq)
    v = real(ifft(v_hat))
    return v
end

function grad(v::Array{Float64, 2}, params::SimulationParams)
    v_hat = fft(v)
    dvx = real(ifft(1im * params.kx .* v_hat))
    dvy = real(ifft(1im * params.ky .* v_hat))
    return dvx, dvy
end

function div(vx::Array{Float64, 2}, vy::Array{Float64, 2}, params::SimulationParams)
    dvx_x = real(ifft(1im * params.kx .* fft(vx)))
    dvy_y = real(ifft(1im * params.ky .* fft(vy)))
    return dvx_x + dvy_y
end

function curl(vx::Array{Float64, 2}, vy::Array{Float64, 2}, params::SimulationParams)
    dvx_y = real(ifft(1im * params.ky .* fft(vx)))
    dvy_x = real(ifft(1im * params.kx .* fft(vy)))
    return dvy_x - dvx_y
end

function apply_dealias(f::Array{Float64, 2}, params::SimulationParams)
    f_hat = params.dealias .* fft(f)
    return real(ifft(f_hat))
end

function time_step(vx::Array{Float64, 2}, vy::Array{Float64, 2}, dt::Float64, nu::Float64, params::SimulationParams)
    dvx_x, dvx_y = grad(vx, params)
    dvy_x, dvy_y = grad(vy, params)

    rhs_x = -(vx .* dvx_x .+ vy .* dvx_y)
    rhs_y = -(vx .* dvy_x .+ vy .* dvy_y)

    rhs_x = apply_dealias(rhs_x, params)
    rhs_y = apply_dealias(rhs_y, params)

    vx .+= dt .* rhs_x
    vy .+= dt .* rhs_y

    div_rhs = div(rhs_x, rhs_y, params)
    P = poisson_solve(div_rhs, params)
    dPx, dPy = grad(P, params)

    vx .-= dt .* dPx
    vy .-= dt .* dPy

    vx = diffusion_solve(vx, dt, nu, params)
    vy = diffusion_solve(vy, dt, nu, params)

    return vx, vy
end

function init_outfile(file_name::String)
    h5file = h5open(file_name, "w")
    close(h5file)
end

function save2file(file_name::String, matrix::Array{Float64, 2}, time::Float64)
    h5file = h5open(file_name, "r+")  # Open file in append mode
    try
        # Create a dataset for the given time with the corresponding array
        dset = create_dataset(h5file, "time_$time", datatype(matrix), size(matrix))
        dset[:, :] = matrix
    finally
        close(h5file)
    end
end

function readfile(file_name::String)
    arrays = Vector{Matrix{Float64}}()
    h5file = h5open(file_name, "r")
    try
        # Get the list of datasets
        datasets = keys(h5file)
        for dataset_name in datasets
            # Read each dataset and append it to the list
            push!(arrays, read(h5file[dataset_name]))
        end
    finally
        close(h5file)
    end
    return arrays
end

function make_gif(file_name::String, arrays::Array{Array{Float64, 2}})
    anim = @animate for array in arrays
        heatmap(array, size=(600, 600), dpi=100, aspect_ratio=1, cmap=:RdBu, clim=(-20, 20), framestyle=:box, legend=false, xticks=[], yticks=[], colorbar=false)
    end
    gif(anim, file_name, fps=20)
end

function main(tEnd::Float64, tOut::Float64)
    N = 256
    L = 1.0

    nu = 0.001
    params = build_params(N, L)

    xlin = LinRange(0, L, N+1)[1:end-1]  # chop off periodic point
    xx, yy = repeat(xlin', N, 1), repeat(xlin, 1, N)

    vx = -sin.(2pi * yy)
    vy = sin.(2pi * xx * 2)

    # Calculate time step based on the stability conditions
    dt = 0.5 * minimum([
        1 / (params.kmax * maximum(vx)),
        1 / (params.kmax * maximum(vy)),
        1 / (params.kmax^2 * nu)
    ])
    
    t = 0.0 
    init_outfile("out.h5")
    wz = curl(vx, vy, params)
    save2file("out.h5", wz, t)
    outputCount = 1

    while t<tEnd
        t += dt; println("t : $t")
        vx, vy = time_step(vx, vy, dt, nu, params)
        
        # Periodic output
        if t >= outputCount * tOut || t >= tEnd
            wz = curl(vx, vy, params)
            save2file("out.h5", wz, t)
            outputCount += 1
        end

        # Recalculate time step
        dt = 0.5 * minimum([
            1 / (params.kmax * maximum(vx)),
            1 / (params.kmax * maximum(vy)),
            1 / (params.kmax^2 * nu)
        ])

        # Check if time step is too small, indicating numerical instability
        if dt < 1e-6
            println("Time step too small, Breaking...")
            break
        end   

    end
    return 

end

main(1.0, 0.01)
arrays = readfile("out.h5")
make_gif("out.gif", arrays)