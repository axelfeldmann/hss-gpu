using HssMatrices

include("hss-gpu.jl")
include("hss-ref.jl") 

K_(x,y) = (x-y) != 0 ? 1/(x-y) : 1.
A = [ K_(x,y) for x=-1:0.005:1, y=-1:0.005:1]
@show size(A)
hssA = hss(A)

B = rand(size(A, 1), 5)

ref = matmatup(hssA, B)

d_hssA = to_gpu(hssA)

# FIXME: most of this should happen on the GPU, but that's not done yet
d_B, d_tmps = prepare_mul(hssA, B)


test_gpu = matmatup_gpu!(d_hssA, d_B, d_tmps)
L2_gpu = unpack(BatchedMats(test_gpu[2]))
@assert L2_gpu[1] ≈ ref.left.data
@assert L2_gpu[2] ≈ ref.right.data