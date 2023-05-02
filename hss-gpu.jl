using CUDA
using LinearAlgebra
using HssMatrices

include("batched-mats.jl")

struct CuHssMat
    # The levelized quantities
    W1::Vector{CuBatchedMats}
    W2::Vector{CuBatchedMats}
    Rs::Vector{CuBatchedMats}
    Bs::Vector{CuBatchedMats}

    # The leaf-only quantities
    D::CuBatchedMats
    U::CuBatchedMats
    V::CuBatchedMats
end

struct CuTmps
    data::Vector{CuBatchedMats}
    F::Vector{CuBatchedMats}
    C::CuBatchedMats
    up_tasks::Vector{CuTask}
    dn_tasks::Vector{CuTask}
end

function get_levels(hssA)
    if isleaf(hssA)
        return 1
    else
        return 1 + max(get_levels(hssA.A11), get_levels(hssA.A22))
    end
end

function to_gpu(hssA::HssMatrix{Float64})::CuHssMat
    num_levels = get_levels(hssA)

    W1 = [ BatchedMats() for _ in 1:num_levels ]
    W2 = [ BatchedMats() for _ in 1:num_levels ]
    Rs = [ BatchedMats() for _ in 1:num_levels ]
    Bs = [ BatchedMats() for _ in 1:num_levels ]

    D = BatchedMats()
    U = BatchedMats()
    V = BatchedMats()

    function fill(hssA::HssMatrix{Float64}, depth::Int64)
        if isleaf(hssA)
            @assert depth == num_levels

            add_mat!(V, transpose(hssA.V))
            add_mat!(U, hssA.U)
            add_mat!(D, hssA.D)
        else
            fill(hssA.A11, depth+1)
            fill(hssA.A22, depth+1)

            add_mat!(W1[depth], transpose(hssA.W1))
            add_mat!(W2[depth], transpose(hssA.W2))
            add_mat!(Rs[depth], hssA.R1)
            add_mat!(Rs[depth], hssA.R2)

            add_mat!(Bs[depth], hssA.B12)
            add_mat!(Bs[depth], hssA.B21)
        end
    end

    fill(hssA, 1)

    return CuHssMat(
        [ CuBatchedMats(b) for b in W1 ],
        [ CuBatchedMats(b) for b in W2 ],
        [ CuBatchedMats(b) for b in Rs ],
        [ CuBatchedMats(b) for b in Bs ],
        CuBatchedMats(D),
        CuBatchedMats(U),
        CuBatchedMats(V)
    )
end

function prepare_mul(hssA::HssMatrix{Float64}, 
                     B::AbstractMatrix{Float64})::Tuple{CuBatchedMats,CuTmps}

    num_levels = get_levels(hssA)

    B_chunks = BatchedMats()
    C_chunks = BatchedMats()
    
    data = [ BatchedMats() for _ in 1:num_levels ]
    F = [ BatchedMats() for _ in 1:num_levels ]

    up_tasks = [ Task(Vector{Int64}(), Vector{Int64}(), Vector{Int64}(), Vector{Int64}()) for _ in 1:(2 * num_levels - 1) ]
    dn_tasks = [ Task(Vector{Int64}(), Vector{Int64}(), Vector{Int64}(), Vector{Int64}()) for _ in 1:(2 * num_levels) ]

    function fill_up_pass(hssA::HssMatrix{Float64}, B::AbstractMatrix{Float64}, depth::Int64)
        if isleaf(hssA)
            add_mat!(B_chunks, B)

            dest_size = (size(hssA.V, 2), size(B, 2))
            add_zero_mat!(data[depth], dest_size)
            dest_idx = data[depth].nmats

            # data[end][i] += V[i] * B[i]... indices must match
            push_blocks!(up_tasks[depth*2 - 1], dest_idx, dest_idx, dest_idx, dest_size)
            return dest_idx, dest_size
        else

            n1 = hssA.sz1[2]
            left_idx, left_size = fill_up_pass(hssA.A11, B[1:n1,:], depth+1)
            right_idx, right_size = fill_up_pass(hssA.A22, B[n1+1:end,:], depth+1)

            dest_size =(size(hssA.W1, 2), left_size[2])
            add_zero_mat!(data[depth], dest_size)
            dest_idx = data[depth].nmats

            # data[depth][i] += W1[i] * data[depth+1][left_idx]
            push_blocks!(up_tasks[depth*2 - 1], dest_idx, dest_idx, left_idx, dest_size)
            # data[depth][i] += W2[i] * data[depth+1][right_idx]
            push_blocks!(up_tasks[depth*2], dest_idx, dest_idx, right_idx, dest_size)
            return dest_idx, dest_size
        end
    end

    function fill_dn_pass(hssA::HssMatrix{Float64}, C::AbstractMatrix{Float64}, depth::Int, F_idx::Union{Int,Nothing})
        if isleaf(hssA)
            add_mat!(C_chunks, C)

            dest_size = size(C)
            dest_idx = C_chunks.nmats

            # C[i] += D[i] * B[i]
            push_blocks!(dn_tasks[depth*2 - 1], dest_idx, dest_idx, dest_idx, dest_size)
            # C[i] += U[i] * F
            push_blocks!(dn_tasks[depth*2], dest_idx, dest_idx, F_idx, dest_size)
        else
            m1, _ = hssA.sz1

            right_data_idx = F[depth].nmats + 2
            left_data_idx = F[depth].nmats + 1

            right_data_size = get_size(data[depth+1], right_data_idx)
            left_data_size = get_size(data[depth+1], left_data_idx)

            @assert size(hssA.B12, 2) == right_data_size[1]
            @assert size(hssA.B21, 2) == left_data_size[1]

            F1_size = (size(hssA.B12, 1), right_data_size[2])
            F2_size = (size(hssA.B21, 1), left_data_size[2])

            F1_idx = add_zero_mat!(F[depth], F1_size)
            F2_idx = add_zero_mat!(F[depth], F2_size)

            # We need to find B and R matrices in the HssA GPU structure.
            # THIS IS JUST A GUESS AS TO WHERE THEY ARE
            R1_idx = F1_idx
            R2_idx = F2_idx

            B12_idx = F1_idx
            B21_idx = F2_idx

            B_task_idx = depth*2 - 1
            R_task_idx = depth*2

            if !isnothing(F_idx)
                push_blocks!(dn_tasks[R_task_idx], F1_idx, R1_idx, F_idx, F1_size)
                push_blocks!(dn_tasks[R_task_idx], F2_idx, R2_idx, F_idx, F2_size)
            end
            push_blocks!(dn_tasks[B_task_idx], F1_idx, B12_idx, right_data_idx, F1_size)
            push_blocks!(dn_tasks[B_task_idx], F2_idx, B21_idx, left_data_idx, F2_size)


            fill_dn_pass(hssA.A11, C[1:m1,:], depth+1, F1_idx)
            fill_dn_pass(hssA.A22, C[m1+1:end,:], depth+1, F2_idx)
        end
    end

    # Populate data structures for up pass
    fill_up_pass(hssA, B, 1)

    # Populate data structures for down pass
    C = zeros(size(hssA,1), size(B,2))
    fill_dn_pass(hssA, C, 1, nothing)

    return CuBatchedMats(B_chunks), CuTmps(
        [ CuBatchedMats(b) for b in data ],
        [ CuBatchedMats(b) for b in F ],
        CuBatchedMats(C_chunks),
        [ CuTask(t) for t in up_tasks ],
        [ CuTask(t) for t in dn_tasks ]
    )
end

function check_dims(task, dest, src1, src2)
    for (d, s1, s2) in zip(task.dest_idx, task.src1_idx, task.src2_idx)
        dest_dims = get_size(dest, d)
        src1_dims = get_size(src1, s1)
        src2_dims = get_size(src2, s2)
        @assert src1_dims[2] == src2_dims[1]
        @assert dest_dims == (src1_dims[1], src2_dims[2])
    end
end

function matmul_gpu!(hssA::CuHssMat, B::CuBatchedMats, tmps::CuTmps)
    mul_accum_gpu!(tmps.data[end], hssA.V, B, tmps.up_tasks[end])

    # Up pass
    for i in (length(tmps.up_tasks)-1):-2:1
        j = i - 1
        dest_lvl = (i + 1) ÷ 2

        mul_accum_gpu!(tmps.data[dest_lvl], hssA.W1[dest_lvl], tmps.data[dest_lvl+1], tmps.up_tasks[j])
        mul_accum_gpu!(tmps.data[dest_lvl], hssA.W2[dest_lvl], tmps.data[dest_lvl+1], tmps.up_tasks[i])
    end

    # Down pass
    for (i, t) in enumerate(tmps.dn_tasks[1:end-2])
        lvl = (i + 1) ÷ 2
        if i % 2 == 0 && !is_empty(t)
            mul_accum_gpu!(tmps.F[lvl], hssA.Rs[lvl], tmps.F[lvl-1], t)
        else
            mul_accum_gpu!(tmps.F[lvl], hssA.Bs[lvl], tmps.data[lvl+1], t)
        end
    end
    mul_accum_gpu!(tmps.C, hssA.D, B, tmps.dn_tasks[end-1])
    mul_accum_gpu!(tmps.C, hssA.U, tmps.F[end-1], tmps.dn_tasks[end])

    return tmps.C
end