include("batched-mats.jl")

using BenchmarkTools
using Plots

nmats = 300
minsize = 10
maxsize = 300

is = rand(minsize:maxsize, nmats)
js = rand(minsize:maxsize, nmats)
ks = rand(minsize:maxsize, nmats)

A_mats = [ rand(i, k) for (i, k) in zip(is, ks) ]
B_mats = [ rand(k, j) for (k, j) in zip(ks, js) ]
C_ref = [ A * B for (A, B) in zip(A_mats, B_mats) ]

A_d = CuBatchedMats(BatchedMats(A_mats))
B_d = CuBatchedMats(BatchedMats(B_mats))
C_d, task = allocate_result(A_d, B_d)

res = mul_accum_gpu!(C_d, A_d, B_d, task)
C_gpu = unpack(BatchedMats(C_d))

for (c_gpu, c_ref) in zip(C_gpu, C_ref)
    @assert c_gpu ≈ c_ref
end
println("Correctness passed")

function benchmark_cpu(A_mats::Vector{Matrix{Float64}}, B_mats::Vector{Matrix{Float64}})
    C_mats = [ zeros(size(A, 1), size(B, 2)) for (A, B) in zip(A_mats, B_mats) ]
    b = @benchmark begin
        for i in 1:lastindex($C_mats)
            mul!($C_mats[i], $A_mats[i], $B_mats[i])
        end
    end evals=1
    return b
end

function gpu_zip_mul_batched(A_mats, B_mats)
    A_d = CuBatchedMats(BatchedMats(A_mats))
    B_d = CuBatchedMats(BatchedMats(B_mats))
    C_d, task = allocate_result(A_d, B_d)

    @show task

    b = @benchmark begin
        mul_accum_gpu!($C_d, $A_d, $B_d, $task)
    end evals=1
    return b
end

function gpu_zip_mul(A_mats::Vector{Matrix{Float64}},
           B_mats::Vector{Matrix{Float64}})

    A_gpu = [CuMatrix(A) for A in A_mats]
    B_gpu = [CuMatrix(B) for B in B_mats]
    C_gpu = [CuMatrix(zeros(size(A, 1), size(B, 2))) for (A, B) in zip(A_mats, B_mats)]

    b = @benchmark begin
        for i in 1:length($A_mats)
            CUDA.CUBLAS.gemm!('N', 'N', 1.0, $A_gpu[i], $B_gpu[i], 0.0, $C_gpu[i])
        end
    end evals = 1
    return b
end

Ns = [ 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000 ]
matsize = [ 16, 32, 64, 128, 256, 512 ]

cpu_speedup = zeros(length(Ns), length(matsize))
batched_speedup = zeros(length(Ns), length(matsize))

for (i, n) in enumerate(Ns)
    for (j, m) in enumerate(matsize)
        @show n, m

        A_mats = [ rand(m, m) for i in 1:n ]
        B_mats = [ rand(m, m) for i in 1:n ]

        gpu_batched = gpu_zip_mul_batched(A_mats, B_mats)
        display(gpu_batched)
        gpu_unbatched = gpu_zip_mul(A_mats, B_mats)
        display(gpu_unbatched)
        cpu = benchmark_cpu(A_mats, B_mats)
        display(cpu)

        cpu_speedup[i, j] = mean(cpu.times) / mean(gpu_batched.times)
        batched_speedup[i, j] = mean(gpu_unbatched.times) / mean(gpu_batched.times)

        @show cpu_speedup[i, j]
        @show batched_speedup[i, j]
    end
end

hm = heatmap(log10.(cpu_speedup),
    xlabel="m",
    ylabel="N",
    xticks=(1:length(matsize), matsize),
    yticks=(1:length(Ns), Ns),
    title="Batched GPU Speedup over CPU",
    fontsize=14,
    colorbar=false,
    c=:coolwarm,
    size=(600, 600),
    dpi=150)

for i in 1:length(Ns)
    for j in 1:length(matsize)
        annotate!(hm, [(j, i, text(round(cpu_speedup[i, j], digits=2), 12, :black))])
    end
end

display(hm)
savefig(hm, "cpu-speedup.png")

hm = heatmap(log10.(batched_speedup),
    xlabel="m",
    ylabel="N",
    xticks=(1:length(matsize), matsize),
    yticks=(1:length(Ns), Ns),
    title="Batched GPU Speedup over Naive GPU",
    fontsize=14,
    colorbar=false,
    c=:coolwarm,
    size=(600, 600),
    dpi=150)

for i in 1:length(Ns)
    for j in 1:length(matsize)
        annotate!(hm, [(j, i, text(round(batched_speedup[i, j], digits=2), 12, :black))])
    end
end

display(hm)
savefig(hm, "gpu-speedup.png")
