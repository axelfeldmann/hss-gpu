using CUDA
using LinearAlgebra

const block_dim = 32

mutable struct Task
    dest_idx::Vector{Int64}
    src1_idx::Vector{Int64}
    src2_idx::Vector{Int64}
    block_row::Vector{Int64}
    block_col::Vector{Int64}
end

struct CuTask
    dest_idx::CuArray{Int64}
    src1_idx::CuArray{Int64}
    src2_idx::CuArray{Int64}
    block_row::CuArray{Int64}
    block_col::CuArray{Int64}
    CuTask(task::Task) = new(CuArray(task.dest_idx), CuArray(task.src1_idx), CuArray(task.src2_idx), 
                             CuArray(task.block_row), CuArray(task.block_col))
end

function Task(t::CuTask)
    return Task(t.dest_idx, t.src1_idx, t.src2_idx, t.block_row, t.block_col)
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

function add_mat!(batch::BatchedMats, mat::Matrix{Float64})
    push!(batch.rows, size(mat, 1))
    push!(batch.cols, size(mat, 2))
    push!(batch.starts, length(batch.data) + 1)
    append!(batch.data, mat)
    batch.nmats += 1
    return batch.nmats
end

function add_zero_mat!(batch::BatchedMats, (rows, cols)::Tuple{Int64, Int64})
    push!(batch.rows, rows)
    push!(batch.cols, cols)
    push!(batch.starts, length(batch.data) + 1)
    for i in 1:(rows * cols)
        push!(batch.data, 0.0)
    end
    batch.nmats += 1
    return batch.nmats
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
    return batch.nmats
end

function push_blocks!(task::Task, dest_idx::Int64, src1_idx::Int64, src2_idx::Int64, (rows, cols)::Tuple{Int64, Int64})

    block_rows = cld(rows, block_dim)
    block_cols = cld(cols, block_dim)

    for r in 1:block_rows
        for c in 1:block_cols
            push!(task.dest_idx, dest_idx)
            push!(task.src1_idx, src1_idx)
            push!(task.src2_idx, src2_idx)
            push!(task.block_row, r)
            push!(task.block_col, c)
        end
    end
end

function get_size(batch::BatchedMats, idx::Int64)
    return batch.rows[idx], batch.cols[idx]
end

function get_size(batch::CuBatchedMats, idx::Int64)
    return batch.rows[idx], batch.cols[idx]
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

function show_dims(b::CuBatchedMats)
    nmats = length(b.rows)
    for i in 1:nmats
        print("(", b.rows[i], ", ", b.cols[i], ")")
        if i < nmats
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

function is_empty(t::CuTask)
    return length(t.dest_idx) == 0
end

function get_idx(start, rows, r, c)
    return start + (c - 1) * rows + (r - 1)
end

function kern(dest_rows, dest_cols, dest_starts, dest_data,
              src1_rows, src1_cols, src1_starts, src1_data,
              src2_rows, src2_cols, src2_starts, src2_data,
              task_dest_idxs, task_src1_idxs, task_src2_idxs, 
              task_block_rows, task_block_cols)

    b = blockIdx().x

    block_row = task_block_rows[b]
    block_col = task_block_cols[b]

    tc = threadIdx().x
    tr = threadIdx().y

    M_idx = task_dest_idxs[b]
    A_idx = task_src1_idxs[b]
    B_idx = task_src2_idxs[b]

    M_start = dest_starts[M_idx]
    M_rows = dest_rows[M_idx]
    M_cols = dest_cols[M_idx]

    A_start = src1_starts[A_idx]
    A_rows = src1_rows[A_idx]
    A_cols = src1_cols[A_idx]

    B_start = src2_starts[B_idx]
    B_rows = src2_rows[B_idx]

    i = (block_row - 1) * blockDim().y + tr
    j = (block_col - 1) * blockDim().x + tc

    @assert A_cols == B_rows

    dest_idx = get_idx(M_start, M_rows, i, j)
    accum = 0

    s_A = @cuStaticSharedMem(eltype(src1_data), (block_dim, block_dim))
    s_B = @cuStaticSharedMem(eltype(src2_data), (block_dim, block_dim))

    num_tiles = cld(A_cols, block_dim)
    for tile in 1:num_tiles
        tile_start_col_A = (tile - 1) * block_dim
        tile_start_row_B = (tile - 1) * block_dim

        s_A_col = tile_start_col_A + tc
        s_B_row = tile_start_row_B + tr

        # Load elements into the shared memory for matrix A
        if i <= A_rows && s_A_col <= A_cols
            s_A[tr, tc] = src1_data[get_idx(A_start, A_rows, i, s_A_col)]
        else
            s_A[tr, tc] = 0
        end

        # Load elements into the shared memory for matrix B
        if s_B_row <= B_rows && j <= M_cols
            s_B[tr, tc] = src2_data[get_idx(B_start, B_rows, s_B_row, j)]
        else
            s_B[tr, tc] = 0
        end

        sync_threads()

        # Perform computation with the loaded tile
        if i <= M_rows && j <= M_cols
            for k in 1:block_dim
                accum += s_A[tr, k] * s_B[k, tc]
            end
        end

        sync_threads()
    end

    if i <= M_rows && j <= M_cols
        dest_data[dest_idx] = accum
    end

    return
end

function mul_accum_gpu!(d_dest::CuBatchedMats, d_src1::CuBatchedMats, d_src2::CuBatchedMats, d_task::CuTask)
    blocks = length(d_task.dest_idx)
    if blocks == 0
        return
    end

    CUDA.@sync begin
        @cuda threads=(block_dim,block_dim) blocks=blocks kern(d_dest.rows_dev, d_dest.cols_dev, d_dest.starts_dev, d_dest.data_dev,
                                                        d_src1.rows_dev, d_src1.cols_dev, d_src1.starts_dev, d_src1.data_dev,
                                                        d_src2.rows_dev, d_src2.cols_dev, d_src2.starts_dev, d_src2.data_dev,
                                                        d_task.dest_idx, d_task.src1_idx, d_task.src2_idx, 
                                                        d_task.block_row, d_task.block_col)
    end
end

function allocate_result(A::CuBatchedMats, B::CuBatchedMats)
    task = Task([], [], [], [], [])
    dest = BatchedMats()
    for (i, (A_rows, A_cols, B_rows, B_cols)) in enumerate(zip(A.rows, A.cols, B.rows, B.cols))
        res_size = (A_rows, B_cols)
        add_zero_mat!(dest, res_size)
        push_blocks!(task, i, i, i, res_size)
    end
    return CuBatchedMats(dest), CuTask(task)
end