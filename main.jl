using HssMatrices
using BenchmarkTools
using Plots
using Folds

include("hss-gpu.jl")

function test_correctness()
    K_(x,y) = (x-y) != 0 ? 1/(x-y) : 1.
    A = [ K_(x,y) for x=-1:0.0005:1, y=-1:0.0005:1]
    @show size(A)
    hssA = hss(A)

    B = rand(size(A, 1), 5)

    ref = hssA * B

    d_hssA = to_gpu(hssA)
    d_B, d_tmps = prepare_mul(hssA, B)

    C_gpu = matmul_gpu!(d_hssA, d_B, d_tmps)
    C_chunks = unpack(BatchedMats(C_gpu))
    C = vcat(C_chunks...)

    @assert C ≈ ref
    println("Test passed")
end

# Check correctness
test_correctness()

function make_hss(A_size)
    K_(x,y) = (x-y) != 0 ? 1/(x-y) : 1.
    A = [ K_(x,y) for x=-1:(2/A_size):1, y=-1:(2/A_size):1]
    @show size(A)
    return hss(A)
end

# Benchmark performance
A_sizes = [ 500, 1000, 2000, 8000, 16000, 32000 ]
hss_mats = Folds.map(make_hss, A_sizes)

bcols = [ 1, 4, 16, 64, 256, 1024 ]
results = zeros(length(A_sizes), length(bcols))

for (i, A_size) in enumerate(A_sizes)

    @show A_size
    hssA = hss_mats[i]

    for (j, b_col) in enumerate(bcols)

        @show b_col

        B = rand(size(hssA, 1), b_col)
        d_hssA = to_gpu(hssA)
        d_B, d_tmps = prepare_mul(hssA, B)

        cpu = @benchmark $hssA * $B
        gpu = @benchmark matmul_gpu!($d_hssA, $d_B, $d_tmps)

        results[i, j] = mean(cpu.times) / mean(gpu.times)
    end
end

@show results

hm = heatmap(log10.(results),
    xlabel="bcols",
    ylabel="A_sizes",
    xticks=(1:length(bcols), bcols),
    yticks=(1:length(A_sizes), A_sizes),
    colorbar_title="Mean Time Ratio",
    title="GPU speedup on A::HssMatrix * B::Matrix",
    fontsize=14,
    colorbar=false,
    c=:coolwarm,
    size=(600, 600),
    dpi=150)

for i in 1:length(A_sizes)
    for j in 1:length(bcols)
        annotate!(hm, [(j, i, text(round(results[i, j], digits=2), 12, :black))])
    end
end

display(hm)
savefig(hm, "results.png")