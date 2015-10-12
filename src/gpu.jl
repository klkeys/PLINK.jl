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
	df          :: DenseArray{Float32,1}, 
	df_buff     :: cl.Buffer, 
	x           :: BEDFile, 
	x_buff      :: cl.Buffer, 
	r           :: DenseArray{Float32,1}, 
	y_buff      :: cl.Buffer, 
	queue       :: cl.CmdQueue, 
	means       :: DenseArray{Float32,1}, 
	m_buff      :: cl.Buffer, 
	invstds     :: DenseArray{Float32,1}, 
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
	cl.copy!(queue, y_buff, sdata(r))
	cl.call(queue, xtyk, (wg_size*y_chunks,p,1), (wg_size,1,1), n32, p32, y_chunks32, blocksize32, wg_size32, x_buff, red_buff, y_buff, m_buff, p_buff, genofloat)
	cl.call(queue, rxtyk, (wg_size,p,1), (wg_size,1,1), n32, y_chunks32, y_blocks32, wg_size32, red_buff, df_buff, genofloat) 
	cl.copy!(queue, sdata(df), df_buff)
	@sync @inbounds @parallel for snp = 1:p2 
		df_x2!(snp, df, x.x2, r, means, invstds, n, x.p)
	end
	cl.fill!(queue, red_buff, 0.0f0)
	return nothing
end
