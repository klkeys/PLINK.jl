# export container object for GPU temporary arrays
export PlinkGPUVariables

type PlinkGPUVariables{T <: Float, V <: cl.Buffer}
    df_buff     :: V 
    x_buff      :: cl.Buffer{Int8}
    y_buff      :: V 
    mask_buff   :: cl.Buffer{Int}
    device      :: cl.Device
    ctx         :: cl.Context
    queue       :: cl.CmdQueue
    m_buff      :: V 
    p_buff      :: V 
    red_buff    :: V 
    xtyk        :: cl.Kernel
    rxtyk       :: cl.Kernel
    reset_x     :: cl.Kernel
    wg_size     :: Int
    y_chunks    :: Int
    r_chunks    :: Int
    n           :: Int
    p           :: Int
    p2          :: Int
    n32         :: Int32
    p32         :: Int32
    y_chunks32  :: Int32
    blocksize32 :: Int32
    wg_size32   :: Int32
    y_blocks32  :: Int32
    r_length32  :: Int32
    genofloat   :: cl.LocalMem{T}
end

function PlinkGPUVariables{T <: Float}(
    df_buff     :: cl.Buffer{T}, 
    x_buff      :: cl.Buffer{Int8},
    y_buff      :: cl.Buffer{T}, 
    mask_buff   :: cl.Buffer{Int},
    device      :: cl.Device,
    ctx         :: cl.Context,
    queue       :: cl.CmdQueue,
    m_buff      :: cl.Buffer{T}, 
    p_buff      :: cl.Buffer{T}, 
    red_buff    :: cl.Buffer{T}, 
    xtyk        :: cl.Kernel,
    rxtyk       :: cl.Kernel,
    reset_x     :: cl.Kernel,
    wg_size     :: Int,
    y_chunks    :: Int,
    r_chunks    :: Int,
    n           :: Int,
    p           :: Int,
    p2          :: Int,
    n32         :: Int32,
    p32         :: Int32,
    y_chunks32  :: Int32,
    blocksize32 :: Int32,
    wg_size32   :: Int32,
    y_blocks32  :: Int32,
    r_length32  :: Int32,
    genofloat   :: cl.LocalMem{T}
)
    PlinkGPUVariables{T, eltype(y_buff)}(df_buff, x_buff, y_buff, mask_buff, device, ctx, queue, m_buff, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
end

function PlinkGPUVariables{T <: Float}(
    z      :: DenseVector{T},
    x      :: BEDFile{T},
    y      :: DenseVector{T},
    kern   :: String      = readall(open(expanduser("/.julia/v0.4/PLINK/src/kernels/iht_kernels64.cl"))),
    mask_n :: DenseVector{Int} = ones(Int, size(y))
)
    n,p         = size(x.geno)
    wg_size     = 512
    y_chunks    = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0)
    y_blocks    = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0)
    r_chunks    = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0)
    wg_size32   = convert(Int32, wg_size)
    n32         = convert(Int32, n)
    p32         = convert(Int32, p)
    y_chunks32  = convert(Int32, y_chunks)
    y_blocks32  = convert(Int32, y_blocks)
    blocksize32 = convert(Int32, x.geno.blocksize)
    r_length32  = convert(Int32, p*y_chunks)
    device      = last(cl.devices(:gpu))
    ctx         = cl.Context(device) :: cl.Context
    queue       = cl.CmdQueue(ctx) :: cl.CmdQueue
    program     = cl.Program(ctx, source=kern) :: cl.Program
    cl.build!(program)
    xtyk        = cl.Kernel(program, "compute_xt_times_vector")
    rxtyk       = cl.Kernel(program, "reduce_xt_vec_chunks")
    reset_x     = cl.Kernel(program, "reset_x")
    x_buff      = cl.Buffer(Int8, ctx, (:r,  :copy), hostbuf = sdata(x.geno.x)) :: cl.Buffer{Int8}
    y_buff      = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(y)) :: cl.Buffer{T}
    m_buff      = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(x.means)) :: cl.Buffer{T} 
    p_buff      = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(x.precs)) :: cl.Buffer{T}
    df_buff     = cl.Buffer(T,    ctx, (:rw, :copy), hostbuf = sdata(z)) :: cl.Buffer{T}
    red_buff    = cl.Buffer(T,    ctx, (:rw), p*y_chunks) :: cl.Buffer{T}
    mask_buff   = cl.Buffer(Int,  ctx, (:r,  :copy), hostbuf = sdata(mask_n)) :: cl.Buffer{Int}
    genofloat   = cl.LocalMem(T, wg_size)

    PlinkGPUVariables{T, cl.Buffer{T}}(df_buff, x_buff, y_buff, mask_buff, device, ctx, queue, m_buff, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, x.geno.n, x.geno.p, x.covar.p, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
end

"""
    df_x2!(snp, df, x, r, mask_n)

This subroutine calculates the gradient for one nongenetic covariate. 

Arguments:

- `snp` is the SNP to use in calculations.
- `df` is the gradient vector.
- `x` is the `BEDFile` object with nongenetic covariates 
- `r` is the vector of residuals.
- `mask_n` is the bitmask on the data.
"""
function df_x2!{T <: Float}(
    snp     :: Int,
    df      :: DenseVector{T},
    x       :: BEDFile{T},
    y       :: DenseVector{T},
    mask_n  :: DenseVector{Int}
)
    m = x.means[x.geno.p+snp]
    s = x.precs[x.geno.p+snp]
    df[x.geno.p+snp] = zero(T)
    @inbounds for case = 1:x.geno.n
        if mask_n[case] == 1
            @fastmath df[x.geno.p+snp] += y[case] * (x.covar.x[case,snp] - m) * s
        end
    end
    return nothing
end

function df_x2!{T <: Float}(
    df      :: DenseVector{T},
    x       :: BEDFile{T},
    y       :: DenseVector{T},
    mask_n  :: DenseVector{Int}
)
    for i = 1:x.covar.p
        df_x2!(i, df, x, y, mask_n)
    end
    return nothing
end

function Base.At_mul_B!{T <: Float}(
    df          :: SharedVector{T},
    df_buff     :: cl.Buffer,
    x           :: BEDFile{T},
    x_buff      :: cl.Buffer,
    y           :: SharedVector{T},
    y_buff      :: cl.Buffer,
    mask_n      :: DenseVector{Int},
    mask_buff   :: cl.Buffer,
    q           :: cl.CmdQueue,
    m_buff      :: cl.Buffer,
    p_buff      :: cl.Buffer,
    red_buff    :: cl.Buffer,
    xtyk        :: cl.Kernel,
    rxtyk       :: cl.Kernel,
    reset_x     :: cl.Kernel,
    wg_size     :: Int,
    y_chunks    :: Int,
    r_chunks    :: Int,
    n           :: Int,
    p           :: Int,
    p2          :: Int,
    n32         :: Int32,
    p32         :: Int32,
    y_chunks32  :: Int32,
    blocksize32 :: Int32,
    wg_size32   :: Int32,
    y_blocks32  :: Int32,
    r_length32  :: Int32,
    genofloat   :: cl.LocalMem
)
    #cl.wait(cl.copy!(queue, y_buff, sdata(y)))
#   cl.fill!(queue, red_buff, zero(T))    # only works with OpenCL 1.2
    #cl.wait(cl.call(queue, reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(T)))
    #cl.wait(cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, mask_buff, genofloat))
    #cl.wait(cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat))
    #cl.wait(cl.copy!(queue, sdata(df), df_buff))
    cl.wait(cl.copy!(q, y_buff, sdata(y)))
    cl.wait(queue(reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(T)))
    cl.wait(queue(xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, mask_buff, genofloat))
    cl.wait(queue(rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat))
    cl.wait(cl.copy!(q, sdata(df), df_buff))
    df_x2!(df, x, y, mask_n)
    return nothing
end

# wrapper functions for At_mul_B!
function copy_y!{T <: Float}(v::PlinkGPUVariables{T}, y::SharedVector{T})
    cl.copy!(v.queue, v.y_buff, sdata(y)) :: cl.NannyEvent
end

function reset_x!{T <: Float}(v::PlinkGPUVariables{T})
    #cl.call(v.queue, v.reset_x, (v.wg_size*v.r_chunks,1,1), nothing, v.red_buff, v.r_length32, zero(T)) :: cl.Event
    queue(v.reset_x, (v.wg_size*v.r_chunks,1,1), nothing, v.red_buff, v.r_length32, zero(T)) :: cl.Event
end

function xty!{T <: Float}(v::PlinkGPUVariables{T})
    #cl.call(v.queue, v.xtyk, (v.wg_size*v.y_chunks, v.p, 1), (v.wg_size, 1, 1), v.n32, v.p32, v.y_chunks32, v.blocksize32, v.wg_size32, v.x_buff, v.red_buff, v.y_buff, v.m_buff, v.p_buff, v.mask_buff, v.genofloat) :: cl.Event
    queue(v.xtyk, (v.wg_size*v.y_chunks, v.p, 1), (v.wg_size, 1, 1), v.n32, v.p32, v.y_chunks32, v.blocksize32, v.wg_size32, v.x_buff, v.red_buff, v.y_buff, v.m_buff, v.p_buff, v.mask_buff, v.genofloat) :: cl.Event
end

function xty_reduce!{T <: Float}(v::PlinkGPUVariables{T})
    #cl.call(v.queue, v.rxtyk, (v.wg_size,v.p,1), (v.wg_size,1,1), v.n32, v.y_chunks32, v.y_blocks32, v.wg_size32, v.red_buff, v.df_buff, v.genofloat) :: cl.Event
    queue(v.rxtyk, (v.wg_size,v.p,1), (v.wg_size,1,1), v.n32, v.y_chunks32, v.y_blocks32, v.wg_size32, v.red_buff, v.df_buff, v.genofloat) :: cl.Event
end

function copy_xty!{T <: Float}(xty::DenseVector{T}, v::PlinkGPUVariables{T})
    cl.copy!(v.queue, sdata(xty), v.df_buff) :: cl.NannyEvent
end

"""
    At_mul_B!(xty, x::BEDFile, y, mask_n, v)

Compute `x' * y` for a `BEDFile` object `x` using a GPU command queue configured in the `PlinkGPUVariables` object `v`.
"""
function Base.At_mul_B!{T <: Float}(
    xty    :: SharedVector{T},
    x      :: BEDFile{T},
    y      :: SharedVector{T},
    mask_n :: DenseVector{Int},
    v      :: PlinkGPUVariables{T}
)
    copy_y!(v, y)
#   cl.fill!(queue, red_buff, zero(T))    # only works with OpenCL 1.2
    reset_x!(v)
    xty!(v)
    xty_reduce!(v)
    copy_xty!(xty, v)
    df_x2!(xty, x, y, mask_n)
    return nothing
end

"""
    At_mul_B(x::BEDFile, y, kernfile, mask_n)

If called with a kernel file, then `At_mul_B()` will attempt to accelerate computations with a GPU.
"""
function Base.At_mul_B{T <: Float}(
    x      :: BEDFile{T},
    y      :: SharedVector{T},
    v      :: PlinkGPUVariables{T};
    mask_n :: DenseVector{Int} = ones(Int, size(y)),
    pids   :: DenseVector{Int} = procs(x),
)
    xty = SharedArray(T, (size(x,2),), pids=pids) :: SharedVector{T}
    At_mul_B!(xty, x, y, v) 
    return xty
end
