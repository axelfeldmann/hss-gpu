using CUDA
using LinearAlgebra
using HssMatrices

thrs_per_block = 1024

mutable struct Task
    dest_idx::Vector{Int64}
    src1_idx::Vector{Int64}
    src2_idx::Vector{Int64}
    block_idx::Vector{Int64}
end

struct CuTask
    dest_idx::CuArray{Int64}
    src1_idx::CuArray{Int64}
    src2_idx::CuArray{Int64}
    block_idx::CuArray{Int64}
    CuTask(task::Task) = new(CuArray(task.dest_idx), CuArray(task.src1_idx), CuArray(task.src2_idx), CuArray(task.block_idx))
end

function Task(t::CuTask)
    return Task(t.dest_idx, t.src1_idx, t.src2_idx, t.block_idx)
end

mutable struct BatchedMats
    nmats::Int
    rows::Vector{Int}
    cols::Vector{Int}
    starts::Vector{Int}
    data::Vector{Float64}
    BatchedMats() = new(0, Int64[], Int64[], Int64[], Float64[])
    BatchedMats(nmats::Int, rows::Vector{Int}, cols::Vector{Int}, starts::Vector{Int}, 
                data::Vector{Float64}) = new(nmats, rows, cols, starts, data)
end

function unpack(batch::BatchedMats)
    matrices = Vector{Matrix{Float64}}(undef, batch.nmats)
    for i in 1:batch.nmats
        nrows = batch.rows[i]
        ncols = batch.cols[i]
        start_idx = batch.starts[i]
        end_idx = start_idx + nrows * ncols - 1
        data = reshape(batch.data[start_idx:end_idx], nrows, ncols)
        matrices[i] = data
    end
    return matrices
end

struct CuBatchedMats
    # On device
    rows_dev::CuArray{Int64}
    cols_dev::CuArray{Int64}
    starts_dev::CuArray{Int64}
    data_dev::CuArray{Float64}

    # On host
    rows::Vector{Int64}
    cols::Vector{Int64}
    starts::Vector{Int64}
end

function CuBatchedMats(batch::BatchedMats)
    return CuBatchedMats(
        CuArray(batch.rows),
        CuArray(batch.cols),
        CuArray(batch.starts),
        CuArray(batch.data),
        batch.rows,
        batch.cols,
        batch.starts
    )
end

function BatchedMats(batch::CuBatchedMats)
    return BatchedMats(
        length(batch.rows),
        batch.rows,
        batch.cols,
        batch.starts,
        Array(batch.data_dev)
    )
end

struct CuHssMat
    levels::Vector{CuBatchedMats}
end

struct CuTmps
    levels::Vector{CuBatchedMats}
    tasks::Vector{CuTask}
end

function get_levels(hssA)
    if isleaf(hssA)
        return 1
    else
        return 1 + max(get_levels(hssA.A11), get_levels(hssA.A22))
    end
end

function BatchedMats(mats::Vector{Matrix{Float64}})
    nmats = length(mats)
    rows = Int64[]
    cols = Int64[]
    starts = Int64[]
    data = Float64[]

    for mat in mats
        push!(rows, size(mat, 1))
        push!(cols, size(mat, 2))
        push!(starts, length(data) + 1)
        append!(data, mat)
    end

    return BatchedMats(nmats, rows, cols, starts, data)
end

function show_dims(b::BatchedMats)
    for i in 1:b.nmats
        print("(", b.rows[i], ", ", b.cols[i], ")")
        if i < b.nmats
            print(", ")
        end
    end
    println()
end

function show_dims(mats::Vector{Matrix{Float64}})
    nmats = length(mats)
    for i in 1:nmats
        print("(", size(mats[i], 1), ", ", size(mats[i], 2), ")")
        if i < nmats
            print(", ")
        end
    end
    println()
end

function add_mat!(batch::BatchedMats, mat::Matrix{Float64})
    push!(batch.rows, size(mat, 1))
    push!(batch.cols, size(mat, 2))
    push!(batch.starts, length(batch.data) + 1)
    append!(batch.data, mat)
    batch.nmats += 1
end

function add_zero_mat!(batch::BatchedMats, (rows, cols)::Tuple{Int64, Int64})
    push!(batch.rows, rows)
    push!(batch.cols, cols)
    push!(batch.starts, length(batch.data) + 1)
    for i in 1:(rows * cols)
        push!(batch.data, 0.0)
    end
    batch.nmats += 1
end

function add_mat!(batch::BatchedMats, t::Transpose{Float64, Matrix{Float64}})

    rows, cols = size(t)

    push!(batch.rows, rows)
    push!(batch.cols, cols)
    push!(batch.starts, length(batch.data) + 1)
    for index in eachindex(t)
        push!(batch.data, t[index])
    end
    batch.nmats += 1
end

function to_gpu(hssA::HssMatrix{Float64})::CuHssMat
    num_levels = get_levels(hssA)
    batches = [ BatchedMats() for _ in 1:(num_levels) ]

    function fill(hssA::HssMatrix{Float64}, depth::Int64)
        if isleaf(hssA)
            @assert depth == num_levels
            add_mat!(batches[depth], transpose(hssA.V))
        else
            fill(hssA.A11, depth+1)
            fill(hssA.A22, depth+1)

            add_mat!(batches[depth], transpose(hssA.W1))
            add_mat!(batches[depth], transpose(hssA.W2))
        end
    end
    fill(hssA, 1)
    return CuHssMat([ CuBatchedMats(b) for b in batches ])
end

function push_blocks!(task::Task, dest_idx::Int64, src1_idx::Int64, src2_idx::Int64, (rows, cols)::Tuple{Int64, Int64})
    num_blocks = cld(rows * cols, thrs_per_block)
    for i in 1:num_blocks
        push!(task.dest_idx, dest_idx)
        push!(task.src1_idx, src1_idx)
        push!(task.src2_idx, src2_idx)
        push!(task.block_idx, i)
    end
end

function allocate_result(A::CuBatchedMats, B::CuBatchedMats)
    task = Task([], [], [], [])
    dest = BatchedMats()
    for (i, (A_rows, A_cols, B_rows, B_cols)) in enumerate(zip(A.rows, A.cols, B.rows, B.cols))
        res_size = (A_rows, B_cols)
        add_zero_mat!(dest, res_size)
        push_blocks!(task, i, i, i, res_size)
    end
    return CuBatchedMats(dest), CuTask(task)
end

function prepare_mul(hssA::HssMatrix{Float64}, 
                     B::AbstractMatrix{Float64})::Tuple{CuBatchedMats,CuTmps}

    leaf_mats = BatchedMats()
    num_levels = get_levels(hssA)
    tasks = [ Task(Vector{Int64}(), Vector{Int64}(), Vector{Int64}(), Vector{Int64}()) for _ in 1:(2 * num_levels - 1) ]
    tmp_batches = [ BatchedMats() for _ in 1:num_levels ]

    function fill(hssA::HssMatrix{Float64}, B::AbstractMatrix{Float64}, depth::Int64)
        if isleaf(hssA)
            # If we're at a leaf, we need to push:
            # - a slice of B into leaf_mats
            # - a destination matrix into tmp_batches
            # - a set of indices into tasks
            add_mat!(leaf_mats, B)
            
            dest_size = (size(hssA.V, 2), size(B, 2))
            add_zero_mat!(tmp_batches[depth], dest_size)
            dest_idx = tmp_batches[depth].nmats

            push_blocks!(tasks[depth * 2 - 1], dest_idx, leaf_mats.nmats, leaf_mats.nmats, dest_size)

            return dest_idx, dest_size
        else
            # If we're not at a leaf, we need to push
            # - a destination matrix into tmp_batches
            # - two sets of indices into two separate tasks
            n1 = hssA.sz1[2]
            left_idx, left_size = fill(hssA.A11, B[1:n1,:], depth+1)
            right_idx, right_size = fill(hssA.A22, B[n1+1:end,:], depth+1)

            dest_size = (size(hssA.W1, 2), left_size[2])
            add_zero_mat!(tmp_batches[depth], dest_size)
            dest_idx = tmp_batches[depth].nmats

            push_blocks!(tasks[depth * 2 - 1], dest_idx, left_idx, left_idx, dest_size)
            push_blocks!(tasks[depth * 2], dest_idx, right_idx, right_idx, dest_size)

            return dest_idx, dest_size
        end
    end

    # Run the function
    fill(hssA, B, 1)

    # Do host-to-device transfers
    d_B = CuBatchedMats(leaf_mats)
    d_tmp_batches = [ CuBatchedMats(b) for b in tmp_batches ]
    d_tasks = [ CuTask(t) for t in tasks ]
    return d_B, CuTmps(d_tmp_batches, d_tasks)
end

function mul_accum_ref(d_dest::CuBatchedMats, d_src1::CuBatchedMats, d_src2::CuBatchedMats, d_task::CuTask)
    @assert false # DEPRECATED DUE TO >1 INDEX PER TASK
    dest = unpack(BatchedMats(d_dest))
    src1 = unpack(BatchedMats(d_src1))
    src2 = unpack(BatchedMats(d_src2))
    task = Task(d_task)

    for (d, s1, s2) in zip(task.dest_idx, task.src1_idx, task.src2_idx)
        dest[d] += src1[s1] * src2[s2] 
    end

    return BatchedMats(dest)
end

function matmatup_cpu_ref!(hssA::CuHssMat, B::CuBatchedMats, tmps::CuTmps)
    @assert false # DEPRECATED DUE TO >1 INDEX PER TASK
    x = mul_accum_ref(tmps.levels[end], hssA.levels[end], B, tmps.tasks[end])
    tmps.levels[end] = CuBatchedMats(x)

    for i in (length(tmps.tasks)-1):-1:1
        lvl = (i + 1) ÷ 2
        x = mul_accum_ref(tmps.levels[lvl], hssA.levels[lvl], tmps.levels[lvl+1], tmps.tasks[i])
        tmps.levels[lvl] = CuBatchedMats(x)
    end
    return tmps.levels
end

function kern(dest_rows, dest_cols, dest_starts, dest_data,
              src1_rows, src1_cols, src1_starts, src1_data,
              src2_rows, src2_cols, src2_starts, src2_data,
              task_dest_idxs, task_src1_idxs, task_src2_idxs, task_block_idxs)

    b = blockIdx().x

    M_task = task_block_idxs[b]

    t = threadIdx().x

    M_idx = task_dest_idxs[b]
    A_idx = task_src1_idxs[b]
    B_idx = task_src2_idxs[b]

    M_pos = (M_task - 1) * 1024 + (t - 1)

    M_start = dest_starts[M_idx]
    M_rows = dest_rows[M_idx]
    M_cols = dest_cols[M_idx]

    A_start = src1_starts[A_idx]
    A_rows = src1_rows[A_idx]
    A_cols = src1_cols[B_idx]

    B_start = src2_starts[B_idx]
    B_rows = src2_rows[B_idx]
    B_cols = src2_cols[B_idx]

    @assert A_cols == B_rows
    if M_pos >= M_rows * M_cols
        return
    end

    j = M_pos ÷ M_rows
    i = M_pos % M_rows
    accum = dest_data[M_start + M_pos]

    for k in 0:(A_cols-1)
        a = src1_data[A_start + k * A_rows + i]
        b = src2_data[B_start + j * B_rows + k]
        accum += a * b
    end

    dest_data[M_start + M_pos] = accum

    return
end

function mul_accum_gpu!(d_dest::CuBatchedMats, d_src1::CuBatchedMats, d_src2::CuBatchedMats, d_task::CuTask)
    blocks = length(d_task.dest_idx)
    if blocks == 0
        return
    end

    CUDA.@sync begin
        @cuda threads=thrs_per_block blocks=blocks kern(d_dest.rows_dev, d_dest.cols_dev, d_dest.starts_dev, d_dest.data_dev,
                                                        d_src1.rows_dev, d_src1.cols_dev, d_src1.starts_dev, d_src1.data_dev,
                                                        d_src2.rows_dev, d_src2.cols_dev, d_src2.starts_dev, d_src2.data_dev,
                                                        d_task.dest_idx, d_task.src1_idx, d_task.src2_idx, d_task.block_idx)
    end
end

function matmatup_gpu!(hssA::CuHssMat, B::CuBatchedMats, tmps::CuTmps)
    mul_accum_gpu!(tmps.levels[end], hssA.levels[end], B, tmps.tasks[end])
    for i in (length(tmps.tasks)-1):-1:1
        lvl = (i + 1) ÷ 2
        mul_accum_gpu!(tmps.levels[lvl], hssA.levels[lvl], tmps.levels[lvl+1], tmps.tasks[i])
    end
    return tmps.levels
end