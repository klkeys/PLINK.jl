# COMPUTE MINOR ALLELE FREQUENCIES
#
# This function calculates the MAF for each SNP of a compressed matrix X in a BEDFile.
#
# Arguments:
# -- x is the BEDFile object containing the compressed n x p design matrix.
#
# Optional Arguments:
# -- y is a temporary array to store a column of X.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function maf(x::BEDFile; y::DenseVector{Float64} = zeros(Float64,x.n))
	z = zeros(Float64,x.p)
	@inbounds for i = 1:x.p
		decompress_genotypes!(y,x,i)
		z[i] = (min( sum(y .== one(Float64)), sum(y .== -one(Float64))) + 0.5*sum(y .== zero(Float64))) / (x.n - sum(isnan(y)))
	end
	return z
end


function maf(x::BEDFile; y::DenseVector{Float32} = zeros(Float32,x.n))
	z = zeros(Float32,x.p)
	@inbounds for i = 1:x.p
		decompress_genotypes!(y,x,i)
		z[i] = (min( sum(y .== one(Float32)), sum(y .== -one(Float32))) + 0.5f0*sum(y .== zero(Float32))) / (x.n - sum(isnan(y)))
	end
	return z
end


 
# SQUARED EUCLIDEAN NORM OF A COLUMN OF A COMPRESSED MATRIX
#
# This function computes the squared L2 (Euclidean) norm of a matrix.
# The squared L2 norm of a vector y = [y1 y2 ... yn] is equal to the sum of its squared components:
#
#     || y ||_2^2 = y1^2 + y2^2 + ... + yn^2.
#
# sumsq() operates on a vector x of compressed genotypes from a PLINK BED file.
# The argument snp chooses the column of genotypes from the uncompressed matrix.
#
# Arguments:
# -- x is the BEDFile object containing the compressed genotypes.
# -- snp is the current SNP (column) to decompress.
# -- n is the number of cases in the uncompressed matrix.
# -- p is the number of predictors in the uncompressed matrix.
# -- blocksize is the number of bytes per column of the uncompressed genotype matrix.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function sumsq_snp(
	x       :: BEDFile, 
	snp     :: Int, 
	means   :: DenseVector{Float64}, 
	invstds :: DenseVector{Float64}
) 
	s = zero(Float64)		# accumulation variable, will eventually equal dot(y,z)
	t = zero(Float64)		# temp variable, output of interpret_genotype
	m = means[snp]
	d = invstds[snp]

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = getindex(x,x.x,case,snp,x.blocksize)
		t = ifelse(isnan(t), zero(Float64), (t - m)*d)
		s += t*t 
	end

	return s
end


function sumsq_snp(
	x       :: BEDFile, 
	snp     :: Int, 
	means   :: DenseVector{Float32}, 
	invstds :: DenseVector{Float32}
) 
	s = zero(Float32)	# accumulation variable, will eventually equal dot(y,z)
	t = zero(Float32) 	# temp variable, output of interpret_genotype
	m = means[snp]
	d = invstds[snp]

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = getindex(x,x.x,case,snp,x.blocksize)
		t = ifelse(isnan(t), zero(Float32), (t - m)*d)
		s += t*t 
	end

	return s
end

function sumsq_covariate(
	x         :: BEDFile, 
	covariate :: Int, 
	means     :: DenseVector{Float64}, 
	invstds   :: DenseVector{Float64}
) 
	t = zero(Float64) 
	s = zero(Float64) 
	m = means[x.p + covariate]
	d = invstds[x.p + covariate]

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = (x.x2[case,covariate] - m) * d 
		s += t*t 
	end

	return s
end

function sumsq_covariate(
	x         :: BEDFile, 
	covariate :: Int, 
	means     :: DenseVector{Float32}, 
	invstds   :: DenseVector{Float32}
) 
	t = zero(Float32) 
	s = zero(Float32) 
	m = means[x.p + covariate]
	d = invstds[x.p + covariate]

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = (x.x2[case,covariate] - m) * d 
		s += t*t 
	end

	return s
end



# SQUARED EUCLIDEAN NORM OF COLUMNS OF A COMPRESSED MATRIX 
#
# Compute the squared L2 norm of each column of a compressed matrix x.
#
# Arguments:
# -- y is the vector to fill with the squared norms.
# -- x is the BEDfile object that contains the compressed n x p design matrix from which to draw the columns.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function sumsq!(
	y       :: SharedVector{Float64}, 
	x       :: BEDFile, 
	means   :: SharedVector{Float64}, 
	invstds :: SharedVector{Float64}
)
	(x.p + x.p2) == length(y) || throw(DimensionMismatch("y must have one row for every column of x"))
	@sync @inbounds @parallel for snp = 1:x.p
		y[snp] = sumsq_snp(x,snp,means,invstds)
	end
	@sync @inbounds @parallel for covariate = 1:x.p2
		y[x.p + covariate] = sumsq_covariate(x,covariate,means,invstds) 
	end

	return nothing 
end

function sumsq!(
	y       :: SharedVector{Float32}, 
	x       :: BEDFile, 
	means   :: SharedVector{Float32}, 
	invstds :: SharedVector{Float32}
)
	(x.p + x.p2) == length(y) || throw(DimensionMismatch("y must have one row for every column of x"))
	@sync @inbounds @parallel for snp = 1:x.p
		y[snp] = sumsq_snp(x,snp,means,invstds)
	end
	@sync @inbounds @parallel for covariate = 1:x.p2
		y[x.p + covariate] = sumsq_covariate(x,covariate,means,invstds) 
	end

	return nothing 
end



function sumsq!(
	y       :: Vector{Float64}, 
	x       :: BEDFile, 
	means   :: Vector{Float64}, 
	invstds :: Vector{Float64}
)
	(x.p + x.p2) == length(y) || throw(DimensionMismatch("y must have one row for every column of x"))
	@inbounds for snp = 1:x.p
		y[snp] = sumsq_snp(x,snp,means,invstds)
	end
	@inbounds for covariate = 1:x.p2
		y[x.p + covariate] = sumsq_covariate(x,covariate,means,invstds) 
	end

	return nothing 
end



function sumsq!(
	y       :: Vector{Float32}, 
	x       :: BEDFile, 
	means   :: Vector{Float32}, 
	invstds :: Vector{Float32}
)
	(x.p + x.p2) == length(y) || throw(DimensionMismatch("y must have one row for every column of x"))
	@inbounds for snp = 1:x.p
		y[snp] = sumsq_snp(x,snp,means,invstds)
	end
	@inbounds for covariate = 1:x.p2
		y[x.p + covariate] = sumsq_covariate(x,covariate,means,invstds) 
	end

	return nothing 
end

# WRAPPER FOR SUMSQ!
#
# Return a vector with the squared L2 norms of the compressed matrix x from a BEDFile.
#
# Arguments:
# -- x is the BEDFile object to use for computing squared L2 norms
#
# Optional Arguments:
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray)
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function sumsq(
	T       :: Type,
	x       :: BEDFile; 
	shared  :: Bool = true, 
	pids    :: DenseVector{Int} = procs(),
	means   :: DenseVector = mean(T, x, shared=shared, pids=pids), 
	invstds :: DenseVector = invstd(x,means, shared=shared, pids=pids)
) 
	y = ifelse(shared, SharedArray(T, x.p + x.p2, init = S -> S[localindexes(S)] = zero(T), pids=pids), zeros(T, x.p + x.p2))
	sumsq!(y,x,means,invstds)
	return y
end


# MEAN OF COLUMNS OF A COMPRESSED MATRIX
#
# Compute the arithmetic means of the columns of a compressed matrix x from a BEDFile. 
# Note that this function will ignore NaNs, unlike the normal Julia function Base.mean.
#
# Arguments:
# -- x is the BEDFile object to use for computing column means
#
# Optional Arguments
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray)
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function mean(T::Type, x::BEDFile; shared::Bool = true, pids::DenseVector{Int} = procs())

	# initialize return vector
	y = ifelse(shared, SharedArray(T, x.p + x.p2, init= S -> S[localindexes(S)] = zero(T), pids=pids), zeros(T, x.p + x.p2))

	@inbounds for snp = 1:x.p
		y[snp] = mean_col(T,x,snp)
	end
	for i = 1:x.p2 
		@inbounds for j = 1:x.n
			y[x.p + i] += x.x2[j,i]
		end
		y[x.p + i] /= x.n
	end
	return y
end

# for mean function, set default type to Float64
mean(x::BEDFile; shared::Bool = true, pids::DenseVector{Int} = procs()) = mean(Float64, x, shared=shared, pids=pids)

function mean_col(T::Type, x::BEDFile, snp::Int)
	s = zero(T)	# accumulation variable, will eventually equal mean(x,col) for current col 
	t = zero(T) # temp variable, output of interpret_genotype
	u = zero(T)	# count the number of people

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = getindex(x,x.x,case,snp,x.blocksize)

		# ensure that we do not count NaNs
		if isfinite(t)
			s += t 
			u += one(T) 
		end
	end
		
	# now divide s by u to get column mean and return
	return s /= u
end

# for previous function, set default type to Float64
mean_col(x::BEDFile, snp::Int) = mean_col(Float64, x, snp)


# INVERSE STANDARD DEVIATION OF COLUMNS OF A COMPRESSED MATRIX
#
# Compute the inverse or reciprocal standard deviations (1 / std) of the columns of a compressed matrix x from a BEDFile. 
# Note that this function will ignore NaNs, unlike the normal Julia function Base.std.
#
# Arguments:
# -- x is the BEDFile object to use for computing column standard deviations 
#
# Optional Arguments
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray)
# -- y is a vector that contains the column means of x. Defaults to PLINK.mean(x).
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
#
### WARNING: need to fix type assertions here!
function invstd(x::BEDFile, means::DenseVector{Float64}; shared::Bool = true, pids::DenseVector{Int} = procs()) 

	# check bounds
	x.p + x.p2 == length(means) || throw(BoundsError("length(means) != size(x,2)"))

	# initialize return vector
	z = ifelse(shared, SharedArray(Float64, x.p + x.p2, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), zeros(Float64, x.p + x.p2))

	@inbounds  for snp = 1:x.p
		z[snp] = invstd_col(Float64, x, snp, means)
	end
	@inbounds for i = 1:x.p2 
		for j = 1:x.n
			z[x.p + i] += (x.x2[j,i] - means[x.p + i])^2
		end
		z[x.p + i] = sqrt((x.n - 1) / z[x.p + i])
	end
	return z
end

function invstd(x::BEDFile, means::DenseVector{Float32}; shared::Bool = true, pids::DenseVector{Int} = procs())  

	# check bounds
	x.p + x.p2 == length(means) || throw(BoundsError("length(means) != size(x,2)"))

	# initialize return vector
	z = ifelse(shared, SharedArray(Float32, x.p + x.p2, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), zeros(Float32, x.p + x.p2))

	@inbounds  for snp = 1:x.p
		z[snp] = invstd_col(Float32, x, snp, means)
	end
	@inbounds for i = 1:x.p2 
		for j = 1:x.n
			z[x.p + i] += (x.x2[j,i] - means[x.p + i])^2
		end
		z[x.p + i] = sqrt((x.n - 1) / z[x.p + i])
	end
	return z
end

# for invstd function, set default type to Float64
#invstd(x::BEDFile; shared::Bool = true, y::DenseVector{Float64} = mean(Float64, x, shared=shared)) = invstd(Float64, x, shared=shared, y=y)


function invstd_col(T::Type, x::BEDFile, snp::Int, means::DenseVector)
	s = zero(T)			# accumulation variable, will eventually equal mean(x,col) for current col 
	t = zero(T) 		# temp variable, output of interpret_genotype
	u = zero(T)			# count the number of people
	m = means[snp]	# mean of current column

	# loop over all n individuals
	@inbounds for case = 1:x.n
		t = getindex(x,x.x,case,snp,x.blocksize)

		# ensure that we do not count NaNs
		if isfinite(t) 
			s        += (t - m)^2 
			u        += one(T) 
		end
	end

	# now compute the std = sqrt(s) / (u - 1))   
	# save inv std in y
	s    = ifelse(s <= zero(T), zero(T), sqrt((u - one(T)) / s)) 
	return s
end


# for previous function, set default bitstype to Float64
invstd_col(x::BEDFile, snp::Int, means::DenseVector) = invstd_col(Float64, x, snp, means)


# COMPUTE THE DOT PRODUCT OF A COLUMN OF X AGAINST Y
#
# This function computes the dot product of a column from the compressed PLINK BED file against a vector y
# of floating point values.
#
# Arguments:
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- y is the vector on which to perform the dot product.
# -- snp is the desired SNP (column) of the decompressed matrix to use for the dot product.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function dot(
	x       :: BEDFile, 
	y       :: DenseVector{Float64}, 
	snp     :: Int, 
	means   :: DenseVector{Float64}, 
	invstds :: DenseVector{Float64}
) 
	s = zero(Float64)				# accumulation variable, will eventually equal dot(y,z)
	m = means[snp]		# mean of SNP predictor
	d = invstds[snp]	# 1/std of SNP predictor

	if snp <= x.p

		# loop over all individuals
		@inbounds for case = 1:x.n
			t = getindex(x,x.x,case,snp,x.blocksize)

			# handle exceptions on t
			t = ifelse(isnan(t), zero(Float64), t - m)

			# accumulate dot product
			s += y[case] * t 
		end
	else
		@inbounds for case = 1:x.n
			s += (x.x2[case,snp-x.p] - m)  * y[case]
		end
	end

	# return the (normalized) dot product 
	return s*d 
end


function dot(
	x       :: BEDFile, 
	y       :: DenseVector{Float32}, 
	snp     :: Int, 
	means   :: DenseVector{Float32}, 
	invstds :: DenseVector{Float32}
) 
	s = zero(Float32)			# accumulation variable, will eventually equal dot(y,z)
	m = means[snp]		# mean of SNP predictor
	d = invstds[snp]	# 1/std of SNP predictor

	if snp <= x.p

		# loop over all individuals
		@inbounds for case = 1:x.n
			t = getindex(x,x.x,case,snp,x.blocksize)

			# handle exceptions on t
			t = ifelse(isnan(t), zero(Float64), t - m)

			# accumulate dot product
			s += y[case] * t 
		end
	else
		@inbounds for case = 1:x.n
			s += (x.x2[case,snp-x.p] - m) * y[case]
		end
	end

	# return the (normalized) dot product 
	return s*d 

end


function dot(
	x       :: BEDFile, 
	y       :: DenseVector{Float64}, 
	snp     :: Int, 
	means   :: DenseVector{Float64}, 
	invstds :: DenseVector{Float64},
	mask_n  :: DenseVector{Int}
) 
	s = zero(Float64)				# accumulation variable, will eventually equal dot(y,z)
	m = means[snp]		# mean of SNP predictor
	d = invstds[snp]	# 1/std of SNP predictor

	if snp <= x.p

		# loop over all individuals
		@inbounds for case = 1:x.n

			# only accumulate if case is not masked
			if mask_n[case] == 1
				t = getindex(x,x.x,case,snp,x.blocksize)

				# handle exceptions on t
				t = ifelse(isnan(t), zero(Float64), t - m)

				# accumulate dot product
				s += y[case] * t 
			end
		end
	else
		@inbounds for case = 1:x.n
			if mask_n[case] == 1
				s += (x.x2[case,snp-x.p] - m)  * y[case]
			end
		end
	end

	# return the (normalized) dot product 
	return s*d 
end


function dot(
	x       :: BEDFile, 
	y       :: DenseVector{Float32}, 
	snp     :: Int, 
	means   :: DenseVector{Float32}, 
	invstds :: DenseVector{Float32},
	mask_n  :: DenseVector{Int}
) 
	s = zero(Float32)			# accumulation variable, will eventually equal dot(y,z)
	m = means[snp]		# mean of SNP predictor
	d = invstds[snp]	# 1/std of SNP predictor

	if snp <= x.p

		# loop over all individuals
		@inbounds for case = 1:x.n

			# only accumulate if case is not masked
			if mask_n[case] == 1
				t = getindex(x,x.x,case,snp,x.blocksize)

				# handle exceptions on t
				t = ifelse(isnan(t), zero(Float32), t - m)

				# accumulate dot product
				s += y[case] * t 
			end
		end
	else
		@inbounds for case = 1:x.n
			if mask_n[case] == 1
				s += (x.x2[case,snp-x.p] - m) * y[case]
			end
		end
	end

	# return the (normalized) dot product 
	return s*d 

end


# DOT PRODUCT ALONG ROWS OF X
#
# This function calculates the dot product of a vector against the rows of a matrix X of genotypes.
# It respects memory stride for column-major array ordering by employing X' in lieu of X.
# Using X' allows us to compute dot(X[i,:], b) as dot(X'[:,i], b) which respects the memory stride.
#
# Arguments:
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- b is the vector to use in the dot product
# -- case is the index of the row of X (column of X') to use in the dot product
# -- indices is a BitArray that indexes the nonzero elements of b
#
# Optional Arguments:
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function dott(
	x       :: BEDFile, 
	b       :: DenseVector{Float64}, 
	case    :: Int, 
	indices :: BitArray{1}, 
	means   :: DenseVector{Float64}, 
	invstds :: DenseVector{Float64}
) 
	s = zero(Float64)		# accumulation variable, will eventually equal dot(y,z)
	t = zero(Float64)		# store interpreted genotype
#	@inbounds for snp = 1:x.p 
	for snp = 1:x.p 

		# if current index of b is FALSE, then skip it since it does not contribute to Xb
		if indices[snp] 

			# decompress genotype, this time from transposed matrix
			t = getindex(x,x.xt,snp,case,x.tblocksize)

			# handle exceptions on t
			t = ifelse(isnan(t), zero(Float64), (t - means[snp]) * invstds[snp])

			# accumulate dot product
			s += b[snp] * t 
		end
	end
#	@inbounds for snp = (x.p+1):(x.p+x.p2)
	for snp = (x.p+1):(x.p+x.p2)
		if indices[snp]
			s += b[snp] * (x.x2t[snp-x.p,case] - means[snp]) * invstds[snp] 
		end
	end

	# return the dot product 
	return s 
end


function dott(
	x       :: BEDFile, 
	b       :: DenseVector{Float32}, 
	case    :: Int, 
	indices :: BitArray{1}, 
	means   :: DenseVector{Float32}, 
	invstds :: DenseVector{Float32}
) 
	s = zero(Float32)		# accumulation variable, will eventually equal dot(y,z)
	t = zero(Float32)		# store interpreted genotype
#	@inbounds for snp = 1:x.p 
	for snp = 1:x.p 

		# if current index of b is FALSE, then skip it since it does not contribute to Xb
		if indices[snp] 

			# decompress genotype, this time from transposed matrix
			t = getindex(x,x.xt,snp,case,x.tblocksize)

			# handle exceptions on t
			t = ifelse(isnan(t), zero(Float32), (t - means[snp]) * invstds[snp])

			# accumulate dot product
			s += b[snp] * t 
		end
	end
#	@inbounds for snp = (x.p+1):(x.p+x.p2)
	for snp = (x.p+1):(x.p+x.p2)
		if indices[snp]
			s += b[snp] * (x.x2t[snp-x.p,case] - means[snp]) * invstds[snp] 
		end
	end

	# return the dot product 
	return s 
end


# PERFORM X * BETA
#
# This function computes the operation X*b for the compressed n x p design matrix X in a BEDFile object in a manner that respects memory stride for column-major arrays.
# It also assumes a sparse b, for which we have an index vector to select the nonzeroes.
#
# Arguments:
# -- Xb is the n-dimensional output vector.
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- b is the p-dimensional vector against which we multiply X.
# -- indices is a BitArray that indexes the nonzeroes in b.
# -- k is the number of nonzeroes to use in computing X*b.
#
# Optional Arguments:
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function xb!(
	Xb      :: DenseVector{Float64}, 
	x       :: BEDFile, 
	b       :: DenseVector{Float64}, 
	indices :: BitArray{1}, 
	k       :: Int; 
	pids    :: DenseVector{Int}     = procs(),
	means   :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids),
)
    # error checking
    0 <= k <= size(x,2) || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
#	k <= sum(indices)   || throw(ArgumentError("k != sum(indices)"))
	k >= sum(indices)   || throw(ArgumentError("Must have k >= sum(indices) or X*b will not compute correctly"))

	# loop over the desired number of predictors 
#	@sync @inbounds @parallel for case = 1:x.n
	for case = 1:x.n
		Xb[case] = dott(x, b, case, indices, means, invstds)	
	end

	return nothing 
end 

function xb!(
	Xb      :: DenseVector{Float32}, 
	x       :: BEDFile, 
	b       :: DenseVector{Float32}, 
	indices :: BitArray{1}, 
	k       :: Int; 
	pids    :: DenseVector{Int}     = procs(),
	means   :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids),
)
    # error checking
    0 <= k <= size(x,2) || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
#	k <= sum(indices)   || throw(ArgumentError("k != sum(indices)"))
	k >= sum(indices)   || throw(ArgumentError("Must have k >= sum(indices) or X*b will not compute correctly"))

	# loop over the desired number of predictors 
#	@sync @inbounds @parallel for case = 1:x.n
	for case = 1:x.n
		Xb[case] = dott(x, b, case, indices, means, invstds)	
	end

	return nothing 
end



function xb!(
	Xb      :: DenseVector{Float64}, 
	x       :: BEDFile, 
	b       :: DenseVector{Float64}, 
	indices :: BitArray{1}, 
	k       :: Int,
	mask_n  :: DenseVector{Int};
	pids    :: DenseVector{Int}     = procs(),
	means   :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids),
	n       :: Int                  = length(Xb)
)
    # error checking
    0 <= k <= size(x,2) || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
#	k <= sum(indices)   || throw(ArgumentError("k != sum(indices)"))
	k >= sum(indices)   || throw(ArgumentError("Must have k >= sum(indices) or X*b will not compute correctly"))

	# loop over the desired number of predictors 
#	@sync @inbounds @parallel for case = 1:x.n
#	@sync @parallel for case = 1:x.n
	for case = 1:x.n
		if mask_n[case] == 1
			Xb[case] = dott(x, b, case, indices, means, invstds)	
		end
	end
	return nothing 
end 

function xb!(
	Xb      :: DenseVector{Float32}, 
	x       :: BEDFile, 
	b       :: DenseVector{Float32}, 
	indices :: BitArray{1}, 
	k       :: Int,
	mask_n  :: DenseVector{Int};
	pids    :: DenseVector{Int}     = procs(),
	means   :: DenseVector{Float32} = mean(Float32,x, shared=true, pids=pids), 
	invstds :: DenseVector{Float32} = invstd(x,means, shared=true, pids=pids),
)
    # error checking
    0 <= k <= size(x,2) || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
#	k <= sum(indices)   || throw(ArgumentError("k != sum(indices)"))
	k >= sum(indices)   || throw(ArgumentError("Must have k >= sum(indices) or X*b will not compute correctly"))

	# loop over the desired number of predictors 
#	@sync @inbounds @parallel for case = 1:x.n
	for case = 1:x.n
		if mask_n[case] == 1
			Xb[case] = dott(x, b, case, indices, means, invstds)	
		end
	end

	return nothing 
end


# WRAPPER FOR XB!
#
# This function allocates, computes, and returns the matrix-vector product X*b using a matrix X of compressed genotypes.
#  
# Arguments:
# -- x is the BEDFile object with the compressed n x p design matrix.
# -- b is the p-dimensional vector against which we multiply X.
# -- k is the number of nonzeroes.
# -- indices is a BitArray that indexes the nonzeroes in b.
#
# Optional Arguments:
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray).
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function xb(
	x       :: BEDFile, 
	b       :: Vector{Float64}, 
	indices :: BitArray{1}, 
	k       :: Int; 
	means   :: Vector{Float64} = mean(Float64,x, shared=false), 
	invstds :: Vector{Float64} = invstd(x,means, shared=false)
) 
	Xb = zeros(Float64,x.n)
	xb!(Xb,x,b,indices,k, means=means, invstds=invstds)
	return Xb
end


function xb(
	x       :: BEDFile, 
	b       :: Vector{Float32}, 
	indices :: BitArray{1}, 
	k       :: Int; 
	means   :: Vector{Float32} = mean(Float32,x, shared=false), 
	invstds :: Vector{Float32} = invstd(x,means, shared=false)
) 
	Xb = zeros(Float32,x.n)
	xb!(Xb,x,b,indices,k, means=means, invstds=invstds)
	return Xb
end

function xb(
	x       :: BEDFile, 
	b       :: SharedVector{Float64}, 
	indices :: BitArray{1}, 
	k       :: Int; 
	pids    :: DenseVector{Int}      = procs(),
	means   :: SharedVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float64} = invstd(x,means, shared=true, pids=pids)
) 
	Xb = SharedArray(Float64, x.n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)
	xb!(Xb,x,b,indices,k, means=means, invstds=invstds)
	return Xb
end


function xb(
	x       :: BEDFile, 
	b       :: SharedVector{Float32}, 
	indices :: BitArray{1}, 
	k       :: Int; 
	pids    :: DenseVector{Int}       = procs(),
	means   :: SharedVector{Float32} = mean(Float32,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float32} = invstd(x,means, shared=true, pids=pids)
) 
	Xb = SharedArray(Float32, x.n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)
	xb!(Xb,x,b,indices,k, means=means, invstds=invstds, pids=pids)
	return Xb
end

function xb(
	x       :: BEDFile, 
	b       :: Vector{Float64}, 
	indices :: BitArray{1}, 
	k       :: Int,
	mask_n  :: DenseVector{Int};
	means   :: Vector{Float64} = mean(Float64,x, shared=false), 
	invstds :: Vector{Float64} = invstd(x,means, shared=false)
) 
	Xb = zeros(Float64,x.n)
	xb!(Xb,x,b,indices,k,mask_n, means=means, invstds=invstds)
	return Xb
end


function xb(
	x       :: BEDFile, 
	b       :: Vector{Float32}, 
	indices :: BitArray{1}, 
	k       :: Int,
	mask_n  :: DenseVector{Int};
	means   :: Vector{Float32} = mean(Float32,x, shared=false), 
	invstds :: Vector{Float32} = invstd(x,means, shared=false)
) 
	Xb = zeros(Float32,x.n)
	xb!(Xb,x,b,indices,k,mask_n, means=means, invstds=invstds)
	return Xb
end

function xb(
	x       :: BEDFile, 
	b       :: SharedVector{Float64}, 
	indices :: BitArray{1}, 
	k       :: Int,
	mask_n  :: DenseVector{Int};
	pids    :: DenseVector{Int}      = procs(),
	means   :: SharedVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float64} = invstd(x,means, shared=true, pids=pids)
) 
	Xb = SharedArray(Float64, x.n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)
	xb!(Xb,x,b,indices,k,mask_n, means=means, invstds=invstds, pids=pids)
	return Xb
end


function xb(
	x       :: BEDFile, 
	b       :: SharedVector{Float32}, 
	indices :: BitArray{1}, 
	k       :: Int,
	mask_n  :: DenseVector{Int};
	pids    :: DenseVector{Int}      = procs(),
	means   :: SharedVector{Float32} = mean(Float32,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float32} = invstd(x,means, shared=true, pids=pids)
) 
	Xb = SharedArray(Float32, x.n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)
	xb!(Xb,x,b,indices,k,mask_n, means=means, invstds=invstds, pids=pids)
	return Xb
end


# PERFORM X'Y, OR A TRANSPOSED MATRIX-VECTOR PRODUCT
#
# This function performs a general matrix-vector multiply with a transposed matrix.
# Compare the results to BLAS.gemv('T', 1.0, X, y, zero(Float64), Xty).
#
# Arguments:
# -- Xty is the output vector to overwrite
# -- x is the BEDfile object that contains the compressed n x p design matrix.
# -- y is the vector used in the matrix-vector multiply
#
# Optional Arguments:
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function xty!(
	Xty     :: SharedVector{Float64}, 
	x       :: BEDFile, 
	y       :: SharedVector{Float64}; 
	pids    :: DenseVector{Int}      = procs(),
	means   :: SharedVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float64} = invstd(x,means, shared=true, pids=pids),
	p       :: Int = size(x,2) 
) 
	# error checking
	p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y) || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
#	@sync @inbounds @parallel for snp = 1:p
	for snp = 1:p
		Xty[snp] = dot(x,y,snp,means,invstds)
	end
	
#    np = length(pids)  # determine the number of processes available
#    i = 1
#    nextidx() = (idx=i; i+=1; idx)
#    @sync begin
#        for pid=1:np
#            if pid != myid() || np == 1
#                @async begin
#                    while true
#                        snp = nextidx()
#                        if snp > p 
#                            break
#                        end
#                        Xty[snp] = remotecall_fetch(pid,dot,x,y,snp,means,invstds)
#                    end
#                end
#            end
#        end
#    end
	return nothing 
end 


function xty!(
	Xty     :: SharedVector{Float32}, 
	x       :: BEDFile, 
	y       :: SharedVector{Float32}; 
	pids    :: DenseVector{Int}       = procs(),
	means   :: SharedVector{Float32} = mean(Float32,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float32} = invstd(x,means, shared=true, pids=pids),
	p       :: Int = size(x,2) 
) 
	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
#	@sync @inbounds @parallel for snp = 1:p
	for snp = 1:p
		Xty[snp] = dot(x,y,snp,means,invstds)
	end
	return nothing 
end 

function xty!(
	Xty     :: Vector{Float64}, 
	x       :: BEDFile, 
	y       :: Vector{Float64}; 
	means   :: Vector{Float64} = mean(Float64,x, shared=false), 
	invstds :: Vector{Float64} = invstd(x,means, shared=false),
	p       :: Int = size(x,2) 
) 
	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@inbounds for snp = 1:p
		Xty[snp] = dot(x,y,snp,means,invstds)
	end
	return nothing 
end 


function xty!(
	Xty     :: Vector{Float32}, 
	x       :: BEDFile, 
	y       :: Vector{Float32}; 
	means   :: Vector{Float32} = mean(Float32,x, shared=false), 
	invstds :: Vector{Float32} = invstd(x,means, shared=false),
	p       :: Int = size(x,2) 
) 
	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@inbounds for snp = 1:p
		Xty[snp] = dot(x,y,snp,means,invstds)
	end
	return nothing 
end 




function xty!(
	Xty     :: SharedVector{Float64}, 
	x       :: BEDFile, 
	y       :: SharedVector{Float64},
	mask_n  :: DenseVector{Int}; 
	pids    :: DenseVector{Int}       = procs(),
	means   :: SharedVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float64} = invstd(x,means, shared=true, pids=pids),
	p       :: Int = size(x,2) 
) 
	# error checking
	p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y) || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
#	@sync @inbounds @parallel for snp = 1:p
	for snp = 1:p
		Xty[snp] = dot(x,y,snp,means,invstds,mask_n)
	end
	return nothing 
end 


function xty!(
	Xty     :: SharedVector{Float32}, 
	x       :: BEDFile, 
	y       :: SharedVector{Float32},
	mask_n  :: DenseVector{Int}; 
	pids    :: DenseVector{Int}       = procs(),
	means   :: SharedVector{Float32} = mean(Float32,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float32} = invstd(x,means, shared=true, pids=pids),
	p       :: Int = size(x,2) 
) 
	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
#	@sync @inbounds @parallel for snp = 1:p
	for snp = 1:p
		Xty[snp] = dot(x,y,snp,means,invstds,mask_n)
	end
	return nothing 
end 

function xty!(
	Xty     :: Vector{Float64}, 
	x       :: BEDFile, 
	y       :: Vector{Float64},
	mask_n  :: DenseVector{Int}; 
	means   :: Vector{Float64} = mean(Float64,x, shared=false), 
	invstds :: Vector{Float64} = invstd(x,means, shared=false),
	p       :: Int = size(x,2) 
) 
	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@inbounds for snp = 1:p
		Xty[snp] = dot(x,y,snp,means,invstds,mask_n)
	end
	return nothing 
end 


function xty!(
	Xty     :: Vector{Float32}, 
	x       :: BEDFile, 
	y       :: Vector{Float32},
	mask_n  :: DenseVector{Int}; 
	means   :: Vector{Float32} = mean(Float32,x, shared=false), 
	invstds :: Vector{Float32} = invstd(x,means, shared=false),
	p       :: Int = size(x,2) 
) 
	# error checking
	x.p <= length(Xty) || throw(ArgumentError("Attempting to fill argument Xty of length $(length(Xty)) with $(x.p) elements!"))
	x.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))

	# loop over the desired number of predictors 
	@inbounds for snp = 1:p
		Xty[snp] = dot(x,y,snp,means,invstds,mask_n)
	end
	return nothing 
end 

# WRAPPER FOR XTY!
#
# This function initializes an output vector and computes x'*y with a BEDFile object.
# Compare output to BLAS.gemv('T', 1.0, x, y).
#
# Arguments:
# -- x is the BEDFile object whose compressed matrix we will decompress in order to compute x'*y.
# -- y is the vector against which we will multiply.
#
# Optional Arguments:
# -- shared is a Bool to indicate whether or not to return a SharedArray. Defaults to true (return SharedArray).
# -- means is a vector of column means for X.
# -- invstds is a vector of reciprocal column standard deviations for X. 
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function xty(
	x       :: BEDFile, 
	y       :: SharedVector{Float64}, 
	mask_n  :: DenseVector{Int}; 
	pids    :: DenseVector{Int}      = procs(),
	means   :: SharedVector{Float64} = mean(Float64,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float64} = invstd(x,means, shared=true, pids=pids)
) 
	p = x.p + x.p2
	Xty = SharedArray(Float64, p, init = S -> S[localindexes(S)] = zero(Float64), pids=pids)
	xty!(Xty,x,y,mask_n, means=means, invstds=invstds, p=p, pids=pids) 
	return Xty
end


function xty(
	x       :: BEDFile, 
	y       :: SharedVector{Float32},
	mask_n  :: DenseVector{Int}; 
	pids    :: DenseVector{Int}      = procs(),
	means   :: SharedVector{Float32} = mean(Float32,x, shared=true, pids=pids), 
	invstds :: SharedVector{Float32} = invstd(x,means, shared=true, pids=pids)
) 
	p = x.p + x.p2
	Xty = SharedArray(Float32, p, init = S -> S[localindexes(S)] = zero(Float32), pids=pids)
	xty!(Xty,x,y,mask_n, means=means, invstds=invstds, p=p) 
	return Xty
end

function xty(
	x       :: BEDFile, 
	y       :: Vector{Float64},
	mask_n  :: DenseVector{Int}; 
	means   :: Vector{Float64} = mean(Float64,x, shared=false), 
	invstds :: Vector{Float64} = invstd(x,means, shared=false)
) 
	p = x.p + x.p2
	Xty = zeros(Float64, p)
	xty!(Xty,x,y,mask_n, means=means, invstds=invstds, p=p) 
	return Xty
end


function xty(
	x       :: BEDFile, 
	y       :: Vector{Float32},
	mask_n  :: DenseVector{Int}; 
	means   :: Vector{Float32} = mean(Float32,x, shared=false), 
	invstds :: Vector{Float32} = invstd(x,means, shared=false)
) 
	p = x.p + x.p2
	Xty = zeros(Float32, p)
	xty!(Xty,x,y,mask_n, means=means, invstds=invstds, p=p) 
	return Xty
end
