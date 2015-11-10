##############################
### DECOMPRESSION ROUTINES ###
##############################



## INDEX A COMPRESSED BEDFILE MATRIX
##
## This subroutine succinctly extracts the dosage at the given case and SNP.
##
## Arguments:
## -- x is the BEDfile object that contains the compressed n x p design matrix X.
## -- case is the index of the current case.
## -- snp is the index of the current SNP.
##
## coded by Kevin L. Keys (2015)
## klkeys@g.ucla.edu
#function getindex(x::BEDFile, case::Int, snp::Int; interpret::Bool = true)
#	genotype_block = x.x[(snp-1)*x.blocksize + iceil(case/4)]
#	k = map_bitshift(case)
#	genotype = (genotype_block >>> k) & THREE8
#	interpret && return interpret_genotype(genotype)
#	return genotype
#end


# GET THE VALUE OF A GENOTYPE IN A COMPRESSED MATRIX
# argument X is almost vacuous because it ensures no conflict with current Array implementations
# it becomes useful for accessing nongenetic covariates
function getindex(
	X         :: BEDFile, 
	x         :: DenseVector{Int8}, 
	row       :: Int, 
	col       :: Int, 
	blocksize :: Int; 
	interpret :: Bool = true,
	float32   :: Bool = false
)
	if col <= X.p
		genotype_block = x[(col-1)*blocksize + ((row - 1) >>> 2) + 1]
		k = 2*((row-1) & 3) 
		genotype = (genotype_block >>> k) & THREE8
        interpret && float32 && return geno32[genotype + ONE8] 
        interpret && return geno64[genotype + ONE8] 
		return genotype
	else
		return X.x2[row,(col-X.p)]
	end
end

# default for getindex with BEDFile, to enable array-like indexing
getindex(x::BEDFile, row::Int, col::Int) = getindex(x, x.x, row, col, x.blocksize, interpret=true, float32=false)

function getindex(x::BEDFile, rowidx::BitArray{1}, colidx::BitArray{1})

	yn = sum(rowidx)
	yp = sum(colidx)
	yblock  = ((yn-1) >>> 2) + 1
	ytblock = ((yp-1) >>> 2) + 1

	y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock) 
	yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock) 
	y2  = x.x2[rowidx,colidx]
	y2t = y2'
	p2  = size(y2,2)

	return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2')
end

function getindex(x::BEDFile, rowidx::UnitRange{Int64}, colidx::BitArray{1})

	yn = length(rowidx)
	yp = sum(colidx)
	yblock  = ((yn-1) >>> 2) + 1
	ytblock = ((yp-1) >>> 2) + 1

	y  = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock) 
	yt = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock) 
	y2 = x.x2[rowidx,colidx]
	p2 = size(y2,2)

	return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2)
end

function getindex(x::BEDFile, rowidx::BitArray{1}, colidx::UnitRange{Int64})

	yn = sum(rowidx)
	yp = length(colidx)
	yblock  = ((yn-1) >>> 2) + 1
	ytblock = ((yp-1) >>> 2) + 1

	y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock) 
	yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock) 
	y2  = x.x2[rowidx,colidx]
	p2  = size(y2,2)
	y2t = y2'

	return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2t)
end


function getindex(x::BEDFile, rowidx::UnitRange{Int64}, colidx::UnitRange{Int64})

	yn = length(rowidx)
	yp = length(colidx)
	yblock  = ((yn-1) >>> 2) + 1
	ytblock = ((yp-1) >>> 2) + 1

	y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock) 
	yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock) 
	y2  = x.x2[rowidx,colidx]
	p2  = size(y2,2)
	y2t = y2'

	return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2t)
end




# DECOMPRESS GENOTYPES FROM INT8 BINARY FORMAT
#
# This function decompresses a column (SNP) of a SNP binary file x. 
# In Julia, these binary files are represented as arrays of Int8 numbers.
# Each SNP genotype is stored in two bits, and all people are assumed to be typed. 
# The genotypes output in y take the centered floating point values -1, 0, and 1.
#
# Arguments:
# -- y is the matrix to fill with (centered) dosages.
# -- x is the BEDfile object that contains the compressed n x p design matrix X.
# -- snp is the current SNP (predictor) to extract.
# -- means is an array of column means for X.
# -- invstds is an array of reciprocal column standard deviations for X.
#
# coded by Kevin L. Keys and Kenneth Lange (2015)
# klkeys@g.ucla.edu
function decompress_genotypes!(
	y       :: DenseVector{Float64}, 
	x       :: BEDFile, 
	snp     :: Int, 
	means   :: DenseVector{Float64}, 
	invstds :: DenseVector{Float64}
)
	m = means[snp]
	d = invstds[snp]
	t = zero(Float64)
	if snp <= x.p
		@inbounds for case = 1:x.n
			t       = getindex(x,x.x,case,snp,x.blocksize) 
			y[case] = ifelse(isnan(t), 0.0, (t - m)*d)
		end
	else
		@inbounds for case = 1:x.n
			y[case] = (x.x2[case,(snp-x.p)] - m) * d
		end
	end
	return nothing 
end


function decompress_genotypes!(
	y       :: DenseVector{Float32}, 
	x       :: BEDFile, 
	snp     :: Int, 
	means   :: DenseVector{Float32}, 
	invstds :: DenseVector{Float32}
)
	m = means[snp]
	d = invstds[snp]
	t = zero(Float32)
	if snp <= x.p
		@inbounds for case = 1:x.n
			t       = getindex(x,x.x,case,snp,x.blocksize, float32=true) 
			y[case] = ifelse(isnan(t), 0.0f0, (t - m)*d)
		end
	else
		@inbounds for case = 1:x.n
			y[case] = (x.x2[case,(snp-x.p)] - m) * d
		end
	end
	return nothing 
end


# WRAPPER FOR DECOMPRESS_GENOTYPES!
#
# This function decompresses from x a column of genotypes corresponding to a single SNP.
# It returns the column of decompressed SNPs.
#
# Arguments:
# -- x is the BEDfile object that contains the compressed n x p design matrix.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function decompress_genotypes(
	x       :: BEDFile, 
	snp     :: Int, 
	means   :: DenseVector{Float64}, 
	invstds :: DenseVector{Float64}; 
	shared  :: Bool = true,
	procs   :: DenseVector{Int} = procs()
)
	y = ifelse(shared, SharedArray(Float64, x.n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), zeros(Float64,x.n))
	decompress_genotypes!(y,x,snp,means,invstds)
	return y 
end


function decompress_genotypes(
	x       :: BEDFile, 
	snp     :: Int, 
	means   :: DenseVector{Float32}, 
	invstds :: DenseVector{Float32}; 
	shared  :: Bool = true,
	pids    :: DenseVector{Int} = procs()
)
	y = ifelse(shared, SharedArray(Float32, x.n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), zeros(Float32,x.n))
	decompress_genotypes!(y,x,snp,means,invstds)
	return y 
end



# DECOMPRESS GENOTYPES FROM PLINK BINARY FORMAT
#
# This function decompresses PLINK BED files into a matrix.
# Use this function to test the accuracy of the linear algebra routines in this module.
# Be VERY careful with this function, since the memory demands from decompressing large portions of x
# can grow quite large.
#
# Arguments:
# -- Y is the matrix to fill with decompressed genotypes.
# -- x is the BEDfile object that contains the compressed n x p design matrix.
#
# Optional Arguments:
# -- y is temporary array for storing a column of genotypes. Defaults to zeros(n).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function decompress_genotypes!(
	Y           :: DenseMatrix{Float64}, 
	x           :: BEDFile; 
	means       :: DenseVector{Float64} = mean(Float64,x), 
	invstds     :: DenseVector{Float64} = invstd(x,means),
	standardize :: Bool = true,
#	pids        :: DenseVector{Int} = procs()
) 

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)
	const xn = size(x,1)
	const xp = size(x,2)

	# ensure dimension compatibility
	n == xn || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= xp || throw(DimensionMismatch("Y has more columns than x"))

	## TODO: parallelize this for SharedArrays
	@inbounds for j = 1:xp
		if standardize
			m = means[j]
			s = invstds[j]
		end
		@inbounds for i = 1:n
			Y[i,j] = getindex(x,x.x,i,j,x.blocksize,interpret=true,float32=false) 
			if standardize
				Y[i,j] = (Y[i,j] - m) * s
			end
		end
	end 

	return nothing 
end


function decompress_genotypes!(
	Y           :: DenseMatrix{Float32}, 
	x           :: BEDFile; 
	means       :: DenseVector{Float32} = mean(Float32,x), 
	invstds     :: DenseVector{Float32} = invstd(x,means),
	standardize :: Bool = true
#	pids        :: DenseVector{Int} = procs()
) 

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)
	const xn = size(x,1)
	const xp = size(x,2)

	# ensure dimension compatibility
	n == xn || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= xp || throw(DimensionMismatch("Y has more columns than x"))

	## TODO: parallelize this for SharedArrays
	@inbounds for j = 1:xp
		if standardize
			m = means[j]
			s = invstds[j]
		end
		@inbounds for i = 1:n
			Y[i,j] = getindex(x,x.x,i,j,x.blocksize,interpret=true,float32=true) 
			if standardize
				Y[i,j] = (Y[i,j] - m) * s
			end
		end
	end 

	return nothing 
end


# DECOMPRESS GENOTYPES FROM PLINK BINARY FORMAT USING INDEX VECTOR
#
# This function decompresses PLINK BED files into a matrix.
# Use this function to test the accuracy of the linear algebra routines in this module.
# Be VERY careful with this function, since the memory demands from decompressing large portions of x
# can grow quite large.
#
# Arguments:
# -- Y is the matrix to fill with decompressed genotypes.
# -- x is the BEDfile object that contains the compressed n x p design matrix.
# -- indices is a BitArray that indexes the columns to use in filling Y.
#
# Optional Arguments:
# -- y is temporary array for storing a column of genotypes. Defaults to zeros(n).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function decompress_genotypes!(
	Y       :: DenseMatrix{Float64}, 
	x       :: BEDFile, 
	indices :: BitArray{1}; 
	means   :: DenseVector{Float64} = mean(Float64,x),
	invstds :: DenseVector{Float64} = invstd(x,means)
)

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)
	const xn = x.n 
	const xp = size(x,2) 

	# ensure dimension compatibility
	n == xn            || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= xp            || throw(DimensionMismatch("Y has more columns than x"))
	sum(indices) <= xp || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0

	quiet = true 
	## TODO: parallelize this for SharedArrays
	@inbounds for snp = 1:xp

		# use this column?
		if indices[snp]

			# add to counter
			current_col += 1
			quiet || println("filling current column $current_col with snp $snp")

			# extract column mean, inv std
			m = means[snp]
			d = invstds[snp]

			if snp <= x.p
				@inbounds for case = 1:n
					t = getindex(x,x.x,case,snp,x.blocksize)
					Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
					quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
				end
			else
				@inbounds for case = 1:n
					Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
					quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
				end
			end

			# quit when Y is filled
			current_col == p && return nothing 
		end
	end 
	return nothing 
end


function decompress_genotypes!(
	Y       :: DenseMatrix{Float32}, 
	x       :: BEDFile, 
	indices :: BitArray{1}; 
	means   :: DenseVector{Float32} = mean(Float32,x),
	invstds :: DenseVector{Float32} = invstd(x,means)
)

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)
	const xn = x.n 
	const xp = size(x,2) 

	# ensure dimension compatibility
	n == xn            || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= xp            || throw(DimensionMismatch("Y has more columns than x"))
	sum(indices) <= xp || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0

	quiet = true 
	## TODO: parallelize this for SharedArrays
	@inbounds for snp = 1:xp

		# use this column?
		if indices[snp]

			# add to counter
			current_col += 1
			quiet || println("filling current column $current_col with snp $snp")

			# extract column mean, inv std
			m = means[snp]
			d = invstds[snp]

			if snp <= x.p
				@inbounds for case = 1:n
					t = getindex(x,x.x,case,snp,x.blocksize, float32=true)
					Y[case,current_col] = ifelse(isnan(t), 0.0f0, (t - m)*d)
					quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
				end
			else
				@inbounds for case = 1:n
					Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
					quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
				end
			end

			# quit when Y is filled
			current_col == p && return nothing 
		end
	end 
	return nothing 
end





function decompress_genotypes!(
	Y       :: DenseMatrix{Float64}, 
	x       :: BEDFile, 
	indices :: BitArray{1},
	mask_n  :: DenseVector{Int}; 
	means   :: DenseVector{Float64} = mean(Float64,x),
	invstds :: DenseVector{Float64} = invstd(x,means)
)

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)
	const xn = x.n 
	const xp = size(x,2) 

	# ensure dimension compatibility
	n <= xn            || throw(DimensionMismatch("column dimension of of Y exceeds column dimension of uncompressed x"))
	n == length(mask_n)   || throw(DimensionMismatch("bitmask mask_n indexes different number of columns than column dimension of Y"))
	p <= xp            || throw(DimensionMismatch("Y has more columns than x"))
	sum(indices) <= xp || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0

	quiet = true 
	@inbounds for snp = 1:xp

		# use this column?
		if indices[snp]

			# add to counter
			current_col += 1
			quiet || println("filling current column $current_col with snp $snp")

			# extract column mean, inv std
			m = means[snp]
			d = invstds[snp]

			if snp <= x.p
				@inbounds for case = 1:n
					if mask_n[case] == 1
						t = getindex(x,x.x,case,snp,x.blocksize)
						Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
						quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
					else
						Y[case,current_col] = 0.0
					end
				end
			else
				@inbounds for case = 1:n
					if mask_n[case] == 1
						Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
						quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
					else
						Y[case,current_col] = 0.0
					end
				end
			end

			# quit when Y is filled
			current_col == p && return nothing 
		end
	end 
	return nothing 
end


function decompress_genotypes!(
	Y       :: DenseMatrix{Float32}, 
	x       :: BEDFile, 
	indices :: BitArray{1},
	mask_n  :: DenseVector{Int}; 
	means   :: DenseVector{Float32} = mean(Float32,x),
	invstds :: DenseVector{Float32} = invstd(x,means)
)

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)
	const xn = x.n 
	const xp = size(x,2) 

	# ensure dimension compatibility
	n <= xn            || throw(DimensionMismatch("column dimension of of Y exceeds column dimension of uncompressed x"))
	n == length(mask_n)   || throw(DimensionMismatch("bitmask mask_n indexes different number of columns than column dimension of Y"))
	p <= xp            || throw(DimensionMismatch("Y has more columns than x"))
	sum(indices) <= xp || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0

	quiet = true 
	@inbounds for snp = 1:xp

		# use this column?
		if indices[snp]

			# add to counter
			current_col += 1
			quiet || println("filling current column $current_col with snp $snp")

			# extract column mean, inv std
			m = means[snp]
			d = invstds[snp]

			if snp <= x.p
				@inbounds for case = 1:n
					if mask_n[case] == 1
						t = getindex(x,x.x,case,snp,x.blocksize, float32=true)
						Y[case,current_col] = ifelse(isnan(t), 0.0f0, (t - m)*d)
						quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
					else
						Y[case,current_col] = 0.0f0
					end
				end
			else
				@inbounds for case = 1:n
					if mask_n[case] == 1
						Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
						quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
					else
						Y[case,current_col] = 0.0f0
					end
				end
			end

			# quit when Y is filled
			current_col == p && return nothing 
		end
	end 
	return nothing 
end


# DECOMPRESS GENOTYPES FROM PLINK BINARY FORMAT USING INDEX VECTOR
#
# This function decompresses PLINK BED files into a matrix.
# Use this function to test the accuracy of the linear algebra routines in this module.
# Be VERY careful with this function, since the memory demands from decompressing large portions of x
# can grow quite large.
#
# Arguments:
# -- Y is the matrix to fill with decompressed genotypes.
# -- x is the BEDfile object that contains the compressed n x p design matrix.
# -- indices is an Int vector that indexes the columns to use in filling Y.
#
# Optional Arguments:
# -- y is temporary array for storing a column of genotypes. Defaults to zeros(n).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function decompress_genotypes!(
	Y       :: DenseMatrix{Float64}, 
	x       :: BEDFile, 
	indices :: DenseVector{Int}; 
	means   :: DenseVector{Float64} = mean(Float64,x), 
	invstds :: DenseVector{Float64} = invstd(x,means)
)

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)
	const xn = x.n
	const xp = size(x,2)

	# ensure dimension compatibility
	n == xn          || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= xp          || throw(DimensionMismatch("Y has more columns than x"))
	length(indices) <= p || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0

	quiet = true 
	@inbounds for snp in indices 

		# add to counter
		current_col += 1
		quiet || println("filling current column $current_col with snp $snp")

		# extract column mean, inv std
		m = means[snp]
		d = invstds[snp]
		if snp <= x.p
			@inbounds for case = 1:n
				t = getindex(x,x.x,case,snp,x.blocksize)
				Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
				quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
			end
		else
			@inbounds for case = 1:n
				Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
			end
		end

		# quit when Y is filled
		current_col == p && return nothing 
	end 

	return nothing 
end


function decompress_genotypes!(
	Y       :: DenseMatrix{Float32}, 
	x       :: BEDFile, 
	indices :: DenseVector{Int}; 
	means   :: DenseVector{Float32} = mean(Float32,x), 
	invstds :: DenseVector{Float32} = invstd(x,means)
)

	# get dimensions of matrix to fill 
	const (n,p) = size(Y)
	const xn = x.n
	const xp = size(x,2)

	# ensure dimension compatibility
	n == xn          || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
	p <= xp          || throw(DimensionMismatch("Y has more columns than x"))
	length(indices) <= p || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

	# counter to ensure that we do not attempt to overfill Y
	current_col = 0

	quiet = true 
	@inbounds for snp in indices 

		# add to counter
		current_col += 1
		quiet || println("filling current column $current_col with snp $snp")

		# extract column mean, inv std
		m = means[snp]
		d = invstds[snp]
		if snp <= x.p
			@inbounds for case = 1:n
				t = getindex(x,x.x,case,snp,x.blocksize)
				Y[case,current_col] = ifelse(isnan(t), 0.0f0, (t - m)*d)
				quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
			end
		else
			@inbounds for case = 1:n
				Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
			end
		end

		# quit when Y is filled
		current_col == p && return nothing 
	end 

	return nothing 

end
