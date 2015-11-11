# shortcut for OpenCL module name
cl = OpenCL

## this subroutine calculates the gradient for the nongenetic covariates in BEDFile x.
#function df_x2!(
#	snp     :: Int, 
#	df      :: DenseVector{Float64}, 
#	x       :: DenseMatrix{Float64}, 
#	r       :: DenseVector{Float64}, 
#	means   :: DenseVector{Float64}, 
#	invstds :: DenseVector{Float64}, 
#	n       :: Int, 
#	p       :: Int
#)
#	m = means[p+snp]
#	s = invstds[p+snp]
#	df[p+snp] = zero(Float64) 
#	 for case = 1:n
#		df[p+snp] += r[case] * (x[case,snp] - m) * s
#	end
#	return nothing
#end


# this subroutine calculates the gradient for the nongenetic covariates in BEDFile x.
function df_x2!(
	snp     :: Int, 
	df      :: DenseVector{Float64}, 
	x       :: DenseMatrix{Float64}, 
	r       :: DenseVector{Float64}, 
	mask_n  :: DenseVector{Int},
	means   :: DenseVector{Float64}, 
	invstds :: DenseVector{Float64}, 
	n       :: Int, 
	p       :: Int
)
	m = means[p+snp]
	s = invstds[p+snp]
	df[p+snp] = zero(Float64) 
	for case = 1:n
		if mask_n[case] == 1
			df[p+snp] += r[case] * (x[case,snp] - m) * s
		end
	end
	return nothing
end

## call this function with a configured GPU command queue
#function xty!(
#	df          :: SharedVector{Float64}, 
#	df_buff     :: cl.Buffer, 
#	x           :: BEDFile, 
#	x_buff      :: cl.Buffer, 
#	y           :: SharedVector{Float64}, 
#	y_buff      :: cl.Buffer, 
#	queue       :: cl.CmdQueue, 
#	means       :: SharedVector{Float64}, 
#	m_buff      :: cl.Buffer, 
#	invstds     :: SharedVector{Float64}, 
#	p_buff      :: cl.Buffer, 
#	red_buff    :: cl.Buffer, 
#	xtyk        :: cl.Kernel, 
#	rxtyk       :: cl.Kernel, 
#	reset_x     :: cl.Kernel,
#	wg_size     :: Int, 
#	y_chunks    :: Int, 
#	r_chunks    :: Int, 
#	n           :: Int, 
#	p           :: Int, 
#	p2          :: Int, 
#	n32         :: Int32, 
#	p32         :: Int32, 
#	y_chunks32  :: Int32,
#	blocksize32 :: Int32, 
#	wg_size32   :: Int32, 
#	y_blocks32  :: Int32, 
#	r_length32  :: Int32, 
#	genofloat   :: cl.LocalMem
#)	
#	cl.copy!(queue, y_buff, sdata(y))
#	cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, genofloat)
#	cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat) 
#	cl.copy!(queue, sdata(df), df_buff)
#	for snp = 1:p2 
#		df_x2!(snp, df, x.x2, y, means, invstds, n, x.p)
#	end
##	cl.fill!(queue, red_buff, zero(Float64))	# only works with OpenCL 1.2
#	cl.call(queue, reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(Float64))
#	return nothing
#end

# variant with a bitmask to skip calculations with specified subjects
# used in crossvalidation
function xty!(
	df          :: SharedVector{Float64}, 
	df_buff     :: cl.Buffer, 
	x           :: BEDFile, 
	x_buff      :: cl.Buffer, 
	y           :: SharedVector{Float64}, 
	y_buff      :: cl.Buffer, 
	mask_n      :: DenseVector{Int},
	mask_buff   :: cl.Buffer,
	queue       :: cl.CmdQueue, 
	means       :: SharedVector{Float64}, 
	m_buff      :: cl.Buffer, 
	invstds     :: SharedVector{Float64}, 
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
	cl.wait(cl.call(queue, reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(Float64)))
	cl.wait(cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, mask_buff, genofloat))
	cl.wait(cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat)) 
	cl.wait(cl.copy!(queue, sdata(df), df_buff))
	for snp = 1:p2 
		df_x2!(snp, df, x.x2, y, mask_n, means, invstds, n, x.p)
	end
	return nothing
end


#function xty!(
#	df          :: Vector{Float64}, 
#	df_buff     :: cl.Buffer, 
#	x           :: BEDFile, 
#	x_buff      :: cl.Buffer, 
#	y           :: Vector{Float64}, 
#	y_buff      :: cl.Buffer, 
#	queue       :: cl.CmdQueue, 
#	means       :: Vector{Float64}, 
#	m_buff      :: cl.Buffer, 
#	invstds     :: Vector{Float64}, 
#	p_buff      :: cl.Buffer, 
#	red_buff    :: cl.Buffer, 
#	xtyk        :: cl.Kernel, 
#	rxtyk       :: cl.Kernel, 
#	reset_x     :: cl.Kernel,
#	wg_size     :: Int, 
#	y_chunks    :: Int, 
#	r_chunks    :: Int, 
#	n           :: Int, 
#	p           :: Int, 
#	p2          :: Int, 
#	n32         :: Int32, 
#	p32         :: Int32, 
#	y_chunks32  :: Int32,
#	blocksize32 :: Int32, 
#	wg_size32   :: Int32, 
#	y_blocks32  :: Int32, 
#	r_lengh32   :: Int32, 
#	genofloat   :: cl.LocalMem
#)	
#	cl.copy!(queue, y_buff, sdata(y))
#	cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, genofloat)
#	cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat) 
#	cl.copy!(queue, sdata(df), df_buff)
#	for snp = 1:p2 
#		df_x2!(snp, df, x.x2, r, means, invstds, n, x.p)
#	end
##	cl.fill!(queue, red_buff, zero(Float64))	# only works with OpenCL 1.2
#	cl.call(queue, reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(Float64))
#	return nothing
#end


function xty!(
	df          :: Vector{Float64}, 
	df_buff     :: cl.Buffer, 
	x           :: BEDFile, 
	x_buff      :: cl.Buffer, 
	y           :: Vector{Float64}, 
	y_buff      :: cl.Buffer, 
	mask_n      :: DenseVector{Int},
	mask_buff   :: cl.Buffer,
	queue       :: cl.CmdQueue, 
	means       :: Vector{Float64}, 
	m_buff      :: cl.Buffer, 
	invstds     :: Vector{Float64}, 
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
	cl.wait(cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, mask_buff, genofloat))
	cl.wait(cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat)) 
	cl.wait(cl.copy!(queue, sdata(df), df_buff))
	for snp = 1:p2 
		df_x2!(snp, df, x.x2, r, mask_n, means, invstds, n, x.p)
	end
#	cl.fill!(queue, red_buff, zero(Float64))	# only works with OpenCL 1.2
	cl.wait(cl.call(queue, reset_x, (wg_size*r_chunks,1,1), nothing, red_buff, r_length32, zero(Float64)))
	return nothing
end



## wrapper for xty!
## for optimal performance, call this function with a configured GPU command queue
#function xty(
#	x           :: BEDFile, 
#	y           :: SharedVector{Float64},
#	kernfile    :: ASCIIString; 
#	means       :: SharedVector{Float64}  = mean(Float64, x), 
#	invstds     :: SharedVector{Float64}  = invstd(x, means), 
#	n           :: Int                    = x.n, 
#	p           :: Int                    = x.p, 
#	p2          :: Int                    = x.p2, 
#	wg_size     :: Int                    = 512,
#	y_chunks    :: Int                    = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
#    y_blocks    :: Int                    = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0), 
#	r_chunks    :: Int                    = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0),
#	wg_size32   :: Int32                  = convert(Int32, wg_size),
#	n32         :: Int32                  = convert(Int32, n),
#	p32         :: Int32                  = convert(Int32, p),
#	y_chunks32  :: Int32                  = convert(Int32, y_chunks),
#	y_blocks32  :: Int32                  = convert(Int32, y_blocks),
#	blocksize32 :: Int32                  = convert(Int32, X.blocksize),
#	r_length32  :: Int32                  = convert(Int32, p*y_chunks),
#	device      :: cl.Device              = last(cl.devices(:gpu)),
#	ctx         :: cl.Context             = cl.Context(device), 
#	queue       :: cl.CmdQueue            = cl.CmdQueue(ctx),
#	program     :: cl.Program             = cl.Program(ctx, source=kernfile) |> cl.build!,
#	xtyk        :: cl.Kernel              = cl.Kernel(program, "compute_xt_times_vector"),
#	rxtyk       :: cl.Kernel              = cl.Kernel(program, "reduce_xt_vec_chunks"),
#	reset_x     :: cl.Kernel              = cl.Kernel(program, "reset_x"),
#	x_buff      :: cl.Buffer              = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
#	y_buff      :: cl.Buffer              = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(r)),
#	m_buff      :: cl.Buffer              = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(means)),
#	p_buff      :: cl.Buffer              = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
#	df_buff     :: cl.Buffer              = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf = sdata(df)),
#	red_buff    :: cl.Buffer              = cl.Buffer(Float64, ctx, (:rw), x.p + p2 * y_chunks),
#	xty_buff    :: cl.Buffer              = cl.Buffer(Float64, ctx, (:rw), x.p + p2), 
#	genofloat   :: cl.LocalMem            = cl.LocalMem(Float64, wg_size)
#)	
#	XtY = SharedArray(Float64, x.p + x.p2, init = S -> S[localindexes(S)] = zero(Float64))
#	xty!(XtY, xty_buff, x, x_buff, y, y_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, wg_size, y_chunks, r_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
#	return XtY 
#end
#
## wrapper for xty!
## for optimal performance, call this function with a configured GPU command queue
#function xty(
#	x           :: BEDFile, 
#	y           :: Vector{Float64},
#	kernfile    :: ASCIIString; 
#	means       :: Vector{Float64}  = mean(Float64, x), 
#	invstds     :: Vector{Float64}  = invstd(x, means), 
#	n           :: Int              = x.n, 
#	p           :: Int              = x.p, 
#	p2          :: Int              = x.p2, 
#	wg_size     :: Int              = 512,
#	y_chunks    :: Int              = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
#    y_blocks    :: Int              = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0), 
#	r_chunks    :: Int              = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0),
#	wg_size32   :: Int32            = convert(Int32, wg_size),
#	n32         :: Int32            = convert(Int32, n),
#	p32         :: Int32            = convert(Int32, p),
#	y_chunks32  :: Int32            = convert(Int32, y_chunks),
#	y_blocks32  :: Int32            = convert(Int32, y_blocks),
#	blocksize32 :: Int32            = convert(Int32, X.blocksize),
#	r_length32  :: Int32            = convert(Int32, p*y_chunks),
#	device      :: cl.Device        = last(cl.devices(:gpu)),
#	ctx         :: cl.Context       = cl.Context(device), 
#	queue       :: cl.CmdQueue      = cl.CmdQueue(ctx),
#	program     :: cl.Program       = cl.Program(ctx, source=kernfile) |> cl.build!,
#	xtyk        :: cl.Kernel        = cl.Kernel(program, "compute_xt_times_vector"),
#	rxtyk       :: cl.Kernel        = cl.Kernel(program, "reduce_xt_vec_chunks"),
#	x_buff      :: cl.Buffer        = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
#	y_buff      :: cl.Buffer        = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(r)),
#	m_buff      :: cl.Buffer        = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(means)),
#	p_buff      :: cl.Buffer        = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
#	df_buff     :: cl.Buffer        = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf = sdata(df)),
#	red_buff    :: cl.Buffer        = cl.Buffer(Float64, ctx, (:rw), p + p2 * y_chunks),
#	xty_buff    :: cl.Buffer        = cl.Buffer(Float64, ctx, (:rw), p + p2), 
#	genofloat   :: cl.LocalMem      = cl.LocalMem(Float64, wg_size)
#)	
#	XtY = zeros(Float64, x.p + x.p2)
#	xty!(XtY, xty_buff, x, x_buff, r, y_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, wg_size, y_chunks, r_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
#	return XtY 
#end


function xty(
	x           :: BEDFile, 
	y           :: SharedVector{Float64},
	kernfile    :: ASCIIString,
	mask_n      :: DenseVector{Int}; 
	means       :: SharedVector{Float64}  = mean(Float64, x), 
	invstds     :: SharedVector{Float64}  = invstd(x, means), 
	n           :: Int                    = x.n, 
	p           :: Int                    = x.p, 
	p2          :: Int                    = x.p2, 
	wg_size     :: Int                    = 512,
	y_chunks    :: Int                    = div(n, wg_size) + (n % wg_size != 0 ? 1 : 0),
    y_blocks    :: Int                    = div(y_chunks, wg_size) + (y_chunks % wg_size != 0 ? 1 : 0), 
	r_chunks    :: Int                    = div(p*y_chunks, wg_size) + ((p*y_chunks) % wg_size != 0 ? 1 : 0),
	wg_size32   :: Int32                  = convert(Int32, wg_size),
	n32         :: Int32                  = convert(Int32, n),
	p32         :: Int32                  = convert(Int32, p),
	y_chunks32  :: Int32                  = convert(Int32, y_chunks),
	y_blocks32  :: Int32                  = convert(Int32, y_blocks),
	blocksize32 :: Int32                  = convert(Int32, X.blocksize),
	r_length32  :: Int32                  = convert(Int32, p*y_chunks),
	device      :: cl.Device              = last(cl.devices(:gpu)),
	ctx         :: cl.Context             = cl.Context(device), 
	queue       :: cl.CmdQueue            = cl.CmdQueue(ctx),
	program     :: cl.Program             = cl.Program(ctx, source=kernfile) |> cl.build!,
	xtyk        :: cl.Kernel              = cl.Kernel(program, "compute_xt_times_vector"),
	rxtyk       :: cl.Kernel              = cl.Kernel(program, "reduce_xt_vec_chunks"),
	reset_x     :: cl.Kernel              = cl.Kernel(program, "reset_x"),
	x_buff      :: cl.Buffer              = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
	y_buff      :: cl.Buffer              = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(r)),
	m_buff      :: cl.Buffer              = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(means)),
	p_buff      :: cl.Buffer              = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
	df_buff     :: cl.Buffer              = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf = sdata(df)),
	red_buff    :: cl.Buffer              = cl.Buffer(Float64, ctx, (:rw), x.p + p2 * y_chunks),
	xty_buff    :: cl.Buffer              = cl.Buffer(Float64, ctx, (:rw), x.p + p2), 
	mask_buff   :: cl.Buffer              = cl.Buffer(Int,     ctx, (:r,  :copy), hostbuf = mask_n),
	genofloat   :: cl.LocalMem            = cl.LocalMem(Float64, wg_size)
)	
	XtY = SharedArray(Float64, x.p + x.p2, init = S -> S[localindexes(S)] = zero(Float64))
	xty!(XtY, xty_buff, x, x_buff, y, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, wg_size, y_chunks, r_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
	return XtY 
end

# wrapper for xty!
# for optimal performance, call this function with a configured GPU command queue
function xty(
	x           :: BEDFile, 
	y           :: Vector{Float64},
	kernfile    :: ASCIIString,
	mask_n      :: DenseVector{Int}; 
	means       :: Vector{Float64}  = mean(Float64, x), 
	invstds     :: Vector{Float64}  = invstd(x, means), 
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
	x_buff      :: cl.Buffer        = cl.Buffer(Int8,    ctx, (:r,  :copy), hostbuf = sdata(X.x)),
	y_buff      :: cl.Buffer        = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(r)),
	m_buff      :: cl.Buffer        = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(means)),
	p_buff      :: cl.Buffer        = cl.Buffer(Float64, ctx, (:r,  :copy), hostbuf = sdata(invstds)),
	df_buff     :: cl.Buffer        = cl.Buffer(Float64, ctx, (:rw, :copy), hostbuf = sdata(df)),
	red_buff    :: cl.Buffer        = cl.Buffer(Float64, ctx, (:rw), p + p2 * y_chunks),
	xty_buff    :: cl.Buffer        = cl.Buffer(Float64, ctx, (:rw), p + p2), 
	mask_buff   :: cl.Buffer        = cl.Buffer(Int,     ctx, (:r,  :copy), hostbuf = mask_n),
	genofloat   :: cl.LocalMem      = cl.LocalMem(Float64, wg_size)
)	
	XtY = zeros(Float64, x.p + x.p2)
	xty!(XtY, xty_buff, x, x_buff, y, y_buff, mask_n, mask_buff, queue, means, m_buff, invstds, p_buff, red_buff, xtyk, rxtyk, wg_size, y_chunks, r_chunks, n, p, x.p2, n32, p32, y_chunks32, blocksize32, wg_size32, y_blocks32, r_length32, genofloat)
	return XtY 
end
