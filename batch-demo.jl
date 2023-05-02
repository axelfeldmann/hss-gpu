include("batched-mats.jl")

using BenchmarkTools

nmats = 300
matsize = 32

A_mats = [ rand(matsize, matsize) for i in 1:nmats ]
B_mats = [ rand(matsize, matsize) for i in 1:nmats ]

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
            $C_mats[i] = $A_mats[i] * $B_mats[i]
        end
    end
    println("\nCPU")
    display(b)
end

function gpu_zip_mul_batched(A_mats, B_mats)
    A_d = CuBatchedMats(BatchedMats(A_mats))
    B_d = CuBatchedMats(BatchedMats(B_mats))
    C_d, task = allocate_result(A_d, B_d)

    b = @benchmark mul_accum_gpu!(C_d, A_d, B_d, task)
    println("\nGPU batched multiply")
    display(b)
end

function gpu_zip_mul(A_mats::Vector{Matrix{Float64}}, 
                     B_mats::Vector{Matrix{Float64}})

    A_mats_d = [ CuMatrix(A) for A in A_mats ]
    B_mats_d = [ CuMatrix(B) for B in B_mats ]
    C_mats_d = [ CuMatrix(zeros(matsize, matsize)) for _ in 1:nmats ]


    b = @benchmark begin
        for i in 1:nmats
            $C_mats_d[i] = $A_mats_d[i] * $B_mats_d[i]
        end
    end
    println("\nGPU zip multiply")
    display(b)
end

benchmark_cpu(A_mats, B_mats)
gpu_zip_mul_batched(A_mats, B_mats)
gpu_zip_mul(A_mats, B_mats)