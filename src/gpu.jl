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
)
    m = x.means[x.geno.p+snp]
    s = x.precs[x.geno.p+snp]
    df[p+snp] = zero(T)
    @inbounds for case = 1:x.geno.n
        if mask_n[case] == 1
            df[p+snp] += r[case] * (x[case,snp] - m) * s
        end
    end
    return nothing
end

"`At_mul_B!()` can also be called with a configured GPU command queue."
function At_mul_B!{T <: Float}(
    df          :: SharedVector{T},
    df_buff     :: cl.Buffer,
    x           :: BEDFile{T},
    x_buff      :: cl.Buffer,
    y           :: SharedVector{T},
    y_buff      :: cl.Buffer,
    mask_n      :: DenseVector{Int},
    mask_buff   :: cl.Buffer,
    queue       :: cl.CmdQueue,
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
    cl.wait(cl.copy!(queue, y_buff, sdata(y)))
#   cl.fill!(queue, red_buff, zero(T))    # only works with OpenCL 1.2
    cl.wait(cl.call(queue, reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(T)))
    cl.wait(cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, mask_buff, genofloat))
    cl.wait(cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat))
    cl.wait(cl.copy!(queue, sdata(df), df_buff))
    @inbounds for snp = 1:p2
        df_x2!(snp, df, x.covar.x, y, mask_n)
    end
    return nothing
end


function At_mul_B!{T <: Float}(
    df          :: Vector{T},
    df_buff     :: cl.Buffer,
    x           :: BEDFile{T},
    x_buff      :: cl.Buffer,
    y           :: Vector{T},
    y_buff      :: cl.Buffer,
    mask_n      :: DenseVector{Int},
    mask_buff   :: cl.Buffer,
    queue       :: cl.CmdQueue,
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
        df_x2!(snp, df, x.covar.x, r, mask_n)
    end
    return nothing
end



"""
    At_mul_B(x::BEDFile, y, kernfile, mask_n)

If called with a kernel file, then `At_mul_B()` will attempt to accelerate computations with a GPU.
"""
function xty{T <: Float}(
    x           :: BEDFile{T},
    y           :: SharedVector{T},
    kernfile    :: AbstractString,
    mask_n      :: DenseVector{Int};
    pids        :: DenseVector{Int} = procs(),
    n           :: Int              = x.geno.n,
    p           :: Int              = x.geno.p,
    p2          :: Int              = x.covar.p,
    wg_size     :: Int              = 512,
    y_chunks    :: Int              = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int              = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0),
    r_chunks    :: Int              = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0),
    wg_size32   :: Int32            = convert(Int32, wg_size),
    n32         :: Int32            = convert(Int32, n),
    p32         :: Int32            = convert(Int32, p),
    y_chunks32  :: Int32            = convert(Int32, y_chunks),
    y_blocks32  :: Int32            = convert(Int32, y_blocks),
    blocksize32 :: Int32            = convert(Int32, x.geno.blocksize),
    r_length32  :: Int32            = convert(Int32, p*y_chunks),
    device      :: cl.Device        = last(cl.devices(:gpu)),
    ctx         :: cl.Context       = cl.Context(device),
    queue       :: cl.CmdQueue      = cl.CmdQueue(ctx),
    program     :: cl.Program       = cl.Program(ctx, source=kernfile) |> cl.build!,
    xtyk        :: cl.Kernel        = cl.Kernel(program, "compute_xt_times_vector"),
    rxtyk       :: cl.Kernel        = cl.Kernel(program, "reduce_xt_vec_chunks"),
    reset_x     :: cl.Kernel        = cl.Kernel(program, "reset_x"),
    x_buff      :: cl.Buffer        = cl.Buffer(Int8, ctx, (:r,  :copy), hostbuf = sdata(x.geno.x)),
    y_buff      :: cl.Buffer        = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(y)),
    m_buff      :: cl.Buffer        = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(x.means)),
    p_buff      :: cl.Buffer        = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(x.precs)),
    red_buff    :: cl.Buffer        = cl.Buffer(T,    ctx, (:rw), x.p + p2*y_chunks),
    xty_buff    :: cl.Buffer        = cl.Buffer(T,    ctx, (:rw), x.p + p2),
    mask_buff   :: cl.Buffer        = cl.Buffer(Int,  ctx, (:r,  :copy), hostbuf = sdata(mask_n)),
    genofloat   :: cl.LocalMem      = cl.LocalMem(T, wg_size)
)
    xty = SharedArray(T, size(x,2), init = S -> S[localindexes(S)] = zero(T), pids=pids)
    At_mul_B!(xty, xty_buff, x, x_buff, y, y_buff, mask_n, mask_buff, queue, m_buff, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, x.covar.p, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
    return xty
end

function xty{T <: Float}(
    x           :: BEDFile{T},
    y           :: Vector{T},
    kernfile    :: AbstractString,
    mask_n      :: DenseVector{Int};
    n           :: Int         = x.geno.n,
    p           :: Int         = x.geno.p,
    p2          :: Int         = x.covar.p,
    wg_size     :: Int         = 512,
    y_chunks    :: Int         = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int         = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0),
    r_chunks    :: Int         = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0),
    wg_size32   :: Int32       = convert(Int32, wg_size),
    n32         :: Int32       = convert(Int32, n),
    p32         :: Int32       = convert(Int32, p),
    y_chunks32  :: Int32       = convert(Int32, y_chunks),
    y_blocks32  :: Int32       = convert(Int32, y_blocks),
    blocksize32 :: Int32       = convert(Int32, x.geno.blocksize),
    r_length32  :: Int32       = convert(Int32, p*y_chunks),
    device      :: cl.Device   = last(cl.devices(:gpu)),
    ctx         :: cl.Context  = cl.Context(device),
    queue       :: cl.CmdQueue = cl.CmdQueue(ctx),
    program     :: cl.Program  = cl.Program(ctx, source=kernfile) |> cl.build!,
    xtyk        :: cl.Kernel   = cl.Kernel(program, "compute_xt_times_vector"),
    rxtyk       :: cl.Kernel   = cl.Kernel(program, "reduce_xt_vec_chunks"),
    reset_x     :: cl.Kernel   = cl.Kernel(program, "reset_x"),
    x_buff      :: cl.Buffer   = cl.Buffer(Int8, ctx, (:r,  :copy), hostbuf = sdata(x.geno.x)),
    y_buff      :: cl.Buffer   = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(y)),
    m_buff      :: cl.Buffer   = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(x.means)),
    p_buff      :: cl.Buffer   = cl.Buffer(T,    ctx, (:r,  :copy), hostbuf = sdata(x.precs)),
    red_buff    :: cl.Buffer   = cl.Buffer(T,    ctx, (:rw), p + p2*y_chunks),
    xty_buff    :: cl.Buffer   = cl.Buffer(T,    ctx, (:rw), p + p2),
    mask_buff   :: cl.Buffer   = cl.Buffer(Int,  ctx, (:r,  :copy), hostbuf = sdata(mask_n)),
    genofloat   :: cl.LocalMem = cl.LocalMem(T, wg_size)
)
    xty = zeros(T, x.p + x.p2)
    At_mul_B!(xty, xty_buff, x, x_buff, y, y_buff, mask_n, mask_buff, queue, m_buff, p_buff, red_buff, xtyk, rxtyk, reset_x, wg_size, y_chunks, r_chunks, n, p, x.covar.p, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
    return xty
end
