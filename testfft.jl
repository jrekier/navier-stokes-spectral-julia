using CUDA
using BenchmarkTools
using FFTW

# inspired by https://discourse.julialang.org/t/unreasonably-fast-fft-on-cuda/109398/8

function try_FFT_on_cpu()
    values = rand(256, 256)
    value_complex = ComplexF32.(values)
    cvalues = similar((value_complex), ComplexF32)
    copyto!(cvalues, values)
    cy = similar(cvalues)
    cF = plan_fft!(cvalues, flags=FFTW.MEASURE)
    @btime a = ($cF*$cy)
    return nothing
end

function try_FFT_on_cuda()
    values = rand(256, 256)
    value_complex = ComplexF32.(values)
    cvalues = similar(cu(value_complex), ComplexF32)
    copyto!(cvalues, values)
    cy = similar(cvalues)
    cF = plan_fft!(cvalues)
    @btime CUDA.@sync a = ($cF*$cy)
    return nothing
end

