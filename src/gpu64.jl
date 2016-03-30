"A shortcut for OpenCL module name."
const cl = OpenCL

"""
    df_x2!(snp, df, x, r, mask_n, means, invstds, n, p)

This subroutine calculates the gradient for one nongenetic covariate for a `BEDFile` object `X`.

Arguments:

- `snp` is the SNP to use in calculations.
- `df` is the gradient vector.
- `x` is the matrix of nongenetic covariates, i.e. `X.x2`.
- `r` is the vector of residuals.
- `mask_n` is the bitmask on the data.
- `means` is a vector of column means of `X`.
- `invstds` is a vector of column precisions of `X`.
- `n` is the number of cases.
- `p` is the number of predictors.
"""
function df_x2!{T <: Float}(
    snp     :: Int,
    df      :: DenseVector{T},
    x       :: DenseMatrix{T},
    r       :: DenseVector{T},
    mask_n  :: DenseVector{Int},
    means   :: DenseVector{T},
    invstds :: DenseVector{T},
    n       :: Int,
    p       :: Int
)
    m = means[p+snp]
    s = invstds[p+snp]
    df[p+snp] = zero(T)
    @inbounds for case = 1:n
        if mask_n[case] == 1
            df[p+snp] += r[case] * (x[case,snp] - m) * s
        end
    end
    return nothing
end

"`xty!()` can also be called with a configured GPU command queue."
function xty!{T <: Float}(
    df          :: SharedVector{T},
    df_buff     :: cl.Buffer,
    x           :: BEDFile,
    x_buff      :: cl.Buffer,
    y           :: SharedVector{T},
    y_buff      :: cl.Buffer,
    mask_n      :: DenseVector{Int},
    mask_buff   :: cl.Buffer,
    queue       :: cl.CmdQueue,
    means       :: SharedVector{T},
    m_buff      :: cl.Buffer,
    invstds     :: SharedVector{T},
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
    cl.wait(cl.copy!(queue, y_buff, sdata(y)))
#   cl.fill!(queue, red_buff, zero(T))    # only works with OpenCL 1.2
    cl.wait(cl.call(queue, reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(T)))
    cl.wait(cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, mask_buff, genofloat))
    cl.wait(cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat))
    cl.wait(cl.copy!(queue, sdata(df), df_buff))
    @inbounds for snp = 1:p2
        df_x2!(snp, df, x.x2, y, mask_n, means, invstds, n, x.p)
    end
    return nothing
end


function xty!{T <: Float}(
    df          :: Vector{T},
    df_buff     :: cl.Buffer,
    x           :: BEDFile,
    x_buff      :: cl.Buffer,
    y           :: Vector{T},
    y_buff      :: cl.Buffer,
    mask_n      :: DenseVector{Int},
    mask_buff   :: cl.Buffer,
    queue       :: cl.CmdQueue,
    means       :: Vector{T},
    m_buff      :: cl.Buffer,
    invstds     :: Vector{T},
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
    r_lengh32   :: Int32,
    genofloat   :: cl.LocalMem
)
    cl.wait(cl.copy!(queue, y_buff, sdata(y)))
#   cl.fill!(queue, red_buff, zero(T))    # only works with OpenCL 1.2
    cl.wait(cl.call(queue, reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(T)))
    cl.wait(cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, mask_buff, genofloat))
    cl.wait(cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat))
    cl.wait(cl.copy!(queue, sdata(df), df_buff))
    @inbounds for snp = 1:p2
        df_x2!(snp, df, x.x2, r, mask_n, means, invstds, n, x.p)
    end
    return nothing
end



"""
    xty(x::BEDFile, y, kernfile, mask_n)

If called with a kernel file, then `xty()` will attempt to accelerate computations with a GPU.
"""
function xty{T <: Float}(
    x           :: BEDFile,
    y           :: SharedVector{T},
    kernfile    :: ASCIIString,
    mask_n      :: DenseVector{Int};
    pids        :: DenseVector{Int} = procs(),
    means       :: SharedVector{T}  = mean(T, x, shared=true, pids=pids),
    invstds     :: SharedVector{T}  = invstd(x, means, shared=true, pids=pids),
    n           :: Int              = x.n,
    p           :: Int              = x.p,
    p2          :: Int              = x.p2,
    wg_size     :: Int              = 512,
    y_chunks    :: Int              = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int              = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0),
    r_chunks    :: Int              = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0),
    wg_size32   :: Int32            = convert(Int32, wg_size),
    n32         :: Int32            = convert(Int32, n),
    p32         :: Int32            = convert(Int32, p),
    y_chunks32  :: Int32            = convert(Int32, y_chunks),
    y_blocks32  :: Int32            = convert(Int32, y_blocks),
    blocksize32 :: Int32            = convert(Int32, X.blocksize),
    r_length32  :: Int32            = convert(Int32, p*y_chunks),
    device      :: cl.Device        = last(cl.devices(:gpu)),
    ctx         :: cl.Context       = cl.Context(device),
    queue       :: cl.CmdQueue      = cl.CmdQueue(ctx),
    program     :: cl.Program       = cl.Program(ctx, source=kernfile) |> cl.build!,
    xtyk        :: cl.Kernel        = cl.Kernel(program, "compute_xt_times_vector"),
    rxtyk       :: cl.Kernel        = cl.Kernel(program, "reduce_xt_vec_chunks"),
    reset_x     :: cl.Kernel        = cl.Kernel(program, "reset_x"),
    x_buff      :: cl.Buffer        = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
    y_buff      :: cl.Buffer        = cl.Buffer(T, ctx, (:r,  :copy), hostbuf = sdata(r)),
    m_buff      :: cl.Buffer        = cl.Buffer(T, ctx, (:r,  :copy), hostbuf = sdata(means)),
    p_buff      :: cl.Buffer        = cl.Buffer(T, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
    red_buff    :: cl.Buffer        = cl.Buffer(T, ctx, (:rw), x.p + p2 * y_chunks),
    xty_buff    :: cl.Buffer        = cl.Buffer(T, ctx, (:rw), x.p + p2),
    mask_buff   :: cl.Buffer        = cl.Buffer(Int,     ctx, (:r,  :copy), hostbuf = sdata(mask_n)),
    genofloat   :: cl.LocalMem      = cl.LocalMem(T, wg_size)
)
    XtY = SharedArray(T, x.p + x.p2, init = S -> S[localindexes(S)] = zero(T), pids=pids)
    xty!(XtY, xty_buff, x, x_buff, y, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
    return XtY
end

function xty{T <: Float}(
    x           :: BEDFile,
    y           :: Vector{T},
    kernfile    :: ASCIIString,
    mask_n      :: DenseVector{Int};
    means       :: Vector{T}   = mean(T, x),
    invstds     :: Vector{T}   = invstd(x, means),
    n           :: Int         = x.n,
    p           :: Int         = x.p,
    p2          :: Int         = x.p2,
    wg_size     :: Int         = 512,
    y_chunks    :: Int         = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int         = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0),
    r_chunks    :: Int         = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0),
    wg_size32   :: Int32       = convert(Int32, wg_size),
    n32         :: Int32       = convert(Int32, n),
    p32         :: Int32       = convert(Int32, p),
    y_chunks32  :: Int32       = convert(Int32, y_chunks),
    y_blocks32  :: Int32       = convert(Int32, y_blocks),
    blocksize32 :: Int32       = convert(Int32, X.blocksize),
    r_length32  :: Int32       = convert(Int32, p*y_chunks),
    device      :: cl.Device   = last(cl.devices(:gpu)),
    ctx         :: cl.Context  = cl.Context(device),
    queue       :: cl.CmdQueue = cl.CmdQueue(ctx),
    program     :: cl.Program  = cl.Program(ctx, source=kernfile) |> cl.build!,
    xtyk        :: cl.Kernel   = cl.Kernel(program, "compute_xt_times_vector"),
    rxtyk       :: cl.Kernel   = cl.Kernel(program, "reduce_xt_vec_chunks"),
    reset_x     :: cl.Kernel   = cl.Kernel(program, "reset_x"),
    x_buff      :: cl.Buffer   = cl.Buffer(Int8,ctx, (:r,  :copy), hostbuf = sdata(X.x)),
    y_buff      :: cl.Buffer   = cl.Buffer(T,   ctx, (:r,  :copy), hostbuf = sdata(r)),
    m_buff      :: cl.Buffer   = cl.Buffer(T,   ctx, (:r,  :copy), hostbuf = sdata(means)),
    p_buff      :: cl.Buffer   = cl.Buffer(T,   ctx, (:r,  :copy), hostbuf = sdata(invstds)),
    red_buff    :: cl.Buffer   = cl.Buffer(T,   ctx, (:rw), p + p2 * y_chunks),
    xty_buff    :: cl.Buffer   = cl.Buffer(T,   ctx, (:rw), p + p2),
    mask_buff   :: cl.Buffer   = cl.Buffer(Int, ctx, (:r,  :copy), hostbuf = sdata(mask_n)),
    genofloat   :: cl.LocalMem = cl.LocalMem(T, wg_size)
)
    XtY = zeros(T, x.p + x.p2)
    xty!(XtY, xty_buff, x, x_buff, y, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
    return XtY
end
