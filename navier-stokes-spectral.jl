using FFTW
using Plots

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Julia)
Ported from Python code by Philip Mocz (2023): https://github.com/pmocz/navier-stokes-spectral-python
by Jeremy Rekier (2024), @jrekier

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""

function poisson_solve(rho, kSq_inv)
    V_hat = -fft(rho) .* kSq_inv
    V = real(ifft(V_hat))
    return V
end

function diffusion_solve(v, dt, nu, kSq)
    v_hat = fft(v) ./ (1.0 .+ dt * nu .* kSq)
    v = real(ifft(v_hat))
    return v
end

function grad(v, kx, ky)
    v_hat = fft(v)
    dvx = real(ifft(1im * kx .* v_hat))
    dvy = real(ifft(1im * ky .* v_hat))
    return dvx, dvy
end

function div(vx, vy, kx, ky)
    dvx_x = real(ifft(1im * kx .* fft(vx)))
    dvy_y = real(ifft(1im * ky .* fft(vy)))
    return dvx_x + dvy_y
end

function curl(vx, vy, kx, ky)
    dvx_y = real(ifft(1im * ky .* fft(vx)))
    dvy_x = real(ifft(1im * kx .* fft(vy)))
    return dvy_x - dvx_y
end

function apply_dealias(f, dealias)
    f_hat = dealias .* fft(f)
    return real(ifft(f_hat))
end

function main()
    N = 400
    t = 0.0
    dt = 0.001
    tOut = 0.01
    tEnd = 1.0
    nu = 0.001
    plotRealTime = true
    
    L = 1.0
    xlin = LinRange(0, L, N+1)[1:end-1]  # chop off periodic point
    xx, yy = repeat(xlin', N, 1), repeat(xlin, 1, N)
    
    vx = -sin.(2pi * yy)
    vy = sin.(2pi * xx * 2)
    
    klin = 2.0 * pi / L * collect(-N/2:N/2-1)
    kmax = maximum(klin)
    kx, ky = repeat(klin', N, 1), repeat(klin, 1, N)
    kx = ifftshift(kx)
    ky = ifftshift(ky)
    kSq = kx.^2 + ky.^2
    kSq_inv = 1.0 ./ kSq
    kSq_inv[kSq .== 0] .= 1
    
    dealias = (abs.(kx) .< (2.0/3.0) * kmax) .& (abs.(ky) .< (2.0/3.0) * kmax)
    
    Nt = ceil(Int, tEnd / dt)
    outputCount = 1
    plt =gr(size=(600, 600), dpi=100, aspect_ratio=1)  # Initial plot setup
    
    for i in 1:Nt
        dvx_x, dvx_y = grad(vx, kx, ky)
        dvy_x, dvy_y = grad(vy, kx, ky)
        
        rhs_x = -(vx .* dvx_x .+ vy .* dvx_y)
        rhs_y = -(vx .* dvy_x .+ vy .* dvy_y)
        
        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)
        
        vx .+= dt .* rhs_x
        vy .+= dt .* rhs_y
        
        div_rhs = div(rhs_x, rhs_y, kx, ky)
        P = poisson_solve(div_rhs, kSq_inv)
        dPx, dPy = grad(P, kx, ky)
        
        vx .-= dt .* dPx
        vy .-= dt .* dPy
        
        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)
        
        wz = curl(vx, vy, kx, ky)
        # println(wz)
        
        t += dt
        println(t)
        
        # Plot in real time
        plotThisTurn = false
        if t + dt > outputCount * tOut
            plotThisTurn = true
        end

        if (plotRealTime && plotThisTurn) || (i == Nt)
            plt = heatmap(xlin, xlin, wz, cmap=:RdBu, clim=(-20, 20), aspect_ratio=:equal, framestyle=:box)
            plot!(legend=false)
            plot!(xticks=[], yticks=[])
            plot!(colorbar=false)
            display(plt)
            outputCount += 1
        end
    end
    savefig(plt, "navier-stokes-spectral.png")
    return 0
end

main()