# shortcut for OpenCL module name
cl = OpenCL

# this subroutine calculates the gradient for the nongenetic covariates in BEDFile x.
@compat function df_x2!(
	snp     :: Int, 
	df      :: DenseArray{Float32,1}, 
	x       :: DenseArray{Float32,2}, 
	r       :: DenseArray{Float32,1}, 
	means   :: DenseArray{Float32,1}, 
	invstds :: DenseArray{Float32,1}, 
	n       :: Int, 
	p       :: Int
)
	m = means[p+snp]
	s = invstds[p+snp]
	df[p+snp] = 0.0f0
	for case = 1:n
		df[p+snp] += r[case] * (x[case,snp] - m) * s
	end
	return nothing
end

# call this function with a configured GPU command queue
@compat function xty!(
	df          :: SharedArray{Float32,1}, 
	df_buff     :: cl.Buffer, 
	x           :: BEDFile, 
	x_buff      :: cl.Buffer, 
	y           :: SharedArray{Float32,1}, 
	y_buff      :: cl.Buffer, 
	queue       :: cl.CmdQueue, 
	means       :: SharedArray{Float32,1}, 
	m_buff      :: cl.Buffer, 
	invstds     :: SharedArray{Float32,1}, 
	p_buff      :: cl.Buffer, 
	red_buff    :: cl.Buffer, 
	xtyk        :: cl.Kernel, 
	rxtyk       :: cl.Kernel, 
	wg_size     :: Int, 
	y_chunks    :: Int, 
	n           :: Int, 
	p           :: Int, 
	p2          :: Int, 
	n32         :: Int32, 
	p32         :: Int32, 
	y_chunks32  :: Int32,
	blocksize32 :: Int32, 
	wg_size32   :: Int32, 
	y_blocks32  :: Int32, 
	genofloat   :: cl.LocalMem
)	
	cl.copy!(queue, y_buff, sdata(y))
	cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, genofloat)
	cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat) 
	cl.copy!(queue, sdata(df), df_buff)
	@sync @inbounds @parallel for snp = 1:p2 
		df_x2!(snp, df, x.x2, y, means, invstds, n, x.p)
	end
	cl.fill!(queue, red_buff, 0.0f0)
	return nothing
end


@compat function xty!(
	df          :: Array{Float32,1}, 
	df_buff     :: cl.Buffer, 
	x           :: BEDFile, 
	x_buff      :: cl.Buffer, 
	y           :: Array{Float32,1}, 
	y_buff      :: cl.Buffer, 
	queue       :: cl.CmdQueue, 
	means       :: Array{Float32,1}, 
	m_buff      :: cl.Buffer, 
	invstds     :: Array{Float32,1}, 
	p_buff      :: cl.Buffer, 
	red_buff    :: cl.Buffer, 
	xtyk        :: cl.Kernel, 
	rxtyk       :: cl.Kernel, 
	wg_size     :: Int, 
	y_chunks    :: Int, 
	n           :: Int, 
	p           :: Int, 
	p2          :: Int, 
	n32         :: Int32, 
	p32         :: Int32, 
	y_chunks32  :: Int32,
	blocksize32 :: Int32, 
	wg_size32   :: Int32, 
	y_blocks32  :: Int32, 
	genofloat   :: cl.LocalMem
)	
	cl.copy!(queue, y_buff, sdata(y))
	cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, genofloat)
	cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat) 
	cl.copy!(queue, sdata(df), df_buff)
	@inbounds for snp = 1:p2 
		df_x2!(snp, df, x.x2, r, means, invstds, n, x.p)
	end
	cl.fill!(queue, red_buff, 0.0f0)
	return nothing
end

@compat function xty(
	x           :: BEDFile, 
	y           :: SharedArray{Float32,1},
	kernfile    :: ASCIIString; 
	means       :: SharedArray{Float32,1} = mean(Float32, x), 
	invstds     :: SharedArray{Float32,1} = invstd(x, means), 
	n           :: Int                    = x.n, 
	p           :: Int                    = x.p, 
	p2          :: Int                    = x.p2, 
	wg_size     :: Int                    = 512,
	y_chunks    :: Int                    = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int                    = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0), 
	wg_size32   :: Int32                  = convert(Int32, wg_size),
	n32         :: Int32                  = convert(Int32, n),
	p32         :: Int32                  = convert(Int32, p),
	y_chunks32  :: Int32                  = convert(Int32, y_chunks),
	y_blocks32  :: Int32                  = convert(Int32, y_blocks),
	blocksize32 :: Int32                  = convert(Int32, X.blocksize),
	device      :: cl.Device              = last(cl.devices(:gpu)),
	ctx         :: cl.Context             = cl.Context(device), 
	queue       :: cl.CmdQueue            = cl.CmdQueue(ctx),
	program     :: cl.Program             = cl.Program(ctx, source=kernfile) |> cl.build!,
	xtyk        :: cl.Kernel              = cl.Kernel(program, "compute_xt_times_vector"),
	rxtyk       :: cl.Kernel              = cl.Kernel(program, "reduce_xt_vec_chunks"),
	x_buff      :: cl.Buffer              = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
	y_buff      :: cl.Buffer              = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(r)),
	m_buff      :: cl.Buffer              = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(means)),
	p_buff      :: cl.Buffer              = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
	df_buff     :: cl.Buffer              = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf = sdata(df)),
	red_buff    :: cl.Buffer              = cl.Buffer(Float32, ctx, (:rw), p + p2 * y_chunks),
	xty_buff    :: cl.Buffer              = cl.Buffer(Float32, ctx, (:rw), p + p2), 
	genofloat   :: cl.LocalMem            = cl.LocalMem(Float32, wg_size)
)	
	XtY = SharedArray(Float32, x.p + x.p2, init = S -> S[localindexes(S)] = 0.0f0)
	xty!(XtY, xty_buff, x, x_buff, y, y_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, wg_size, y_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, genofloat)
	return XtY 
end

# wrapper for xty!
# for optimal performance, call this function with a configured GPU command queue
@compat function xty(
	x           :: BEDFile, 
	y           :: Array{Float32,1},
	kernfile    :: ASCIIString; 
	means       :: Array{Float32,1} = mean(Float32, x), 
	invstds     :: Array{Float32,1} = invstd(x, means), 
	n           :: Int              = x.n, 
	p           :: Int              = x.p, 
	p2          :: Int              = x.p2, 
	wg_size     :: Int              = 512,
	y_chunks    :: Int              = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int              = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0), 
	wg_size32   :: Int32            = convert(Int32, wg_size),
	n32         :: Int32            = convert(Int32, n),
	p32         :: Int32            = convert(Int32, p),
	y_chunks32  :: Int32            = convert(Int32, y_chunks),
	y_blocks32  :: Int32            = convert(Int32, y_blocks),
	blocksize32 :: Int32            = convert(Int32, X.blocksize),
	device      :: cl.Device        = last(cl.devices(:gpu)),
	ctx         :: cl.Context       = cl.Context(device), 
	queue       :: cl.CmdQueue      = cl.CmdQueue(ctx),
	program     :: cl.Program       = cl.Program(ctx, source=kernfile) |> cl.build!,
	xtyk        :: cl.Kernel        = cl.Kernel(program, "compute_xt_times_vector"),
	rxtyk       :: cl.Kernel        = cl.Kernel(program, "reduce_xt_vec_chunks"),
	x_buff      :: cl.Buffer        = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
	y_buff      :: cl.Buffer        = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(r)),
	m_buff      :: cl.Buffer        = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(means)),
	p_buff      :: cl.Buffer        = cl.Buffer(Float32, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
	df_buff     :: cl.Buffer        = cl.Buffer(Float32, ctx, (:rw, :copy), hostbuf = sdata(df)),
	red_buff    :: cl.Buffer        = cl.Buffer(Float32, ctx, (:rw), p + p2 * y_chunks),
	xty_buff    :: cl.Buffer        = cl.Buffer(Float32, ctx, (:rw), p + p2), 
	genofloat   :: cl.LocalMem      = cl.LocalMem(Float32, wg_size)
)	
	XtY = zeros(Float32, x.p + x.p2)
	xty!(XtY, xty_buff, x, x_buff, r, y_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, wg_size, y_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, genofloat)
	return XtY 
end


