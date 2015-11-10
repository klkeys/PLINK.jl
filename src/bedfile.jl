##########################################
### CONSTRUCTORS AND UTILITY FUNCTIONS ###
##########################################

# The BEDFile type encodes the vector of compressed genotypes.
# It also encodes the number of samples and the number of predictors in the UNcompressed matrix.
# Finally, it encodes the blocksize for decompression purposes.
# These four features uniquely define any compressed genotype matrix. 
# Note that this BEDFile object, and the rest of this module for that matter, operate with the assumption
# that the compressed matrix is in column-major (SNP-major) format.
# Row-major (case-major) format is not supported.
type BEDFile
	x   :: SharedVector{Int8}	# compressed genotypes for genotype matrix X
	xt  :: SharedVector{Int8}	# compressed genotypes for TRANSPOSED genotype matrix X'
	n   :: Int             	    # number of cases (people) in uncompressed genotype matrix 
	p   :: Int             	    # number of predictors (SNPs) in uncompressed genotype matrix
	blocksize  :: Int     	    # number of bytes per compressed column of genotype matrix
	tblocksize :: Int    	    # number of bytes per compressed column of TRANSPOSED genotype matrix
	x2  :: SharedArray			# nongenetic covariantes, if any exist
	p2  :: Int				    # number of nongenetic covariates
	x2t :: SharedArray			# transpose of nongenetic covariantes, used in matrix algebra 

	BEDFile(x,xt,n,p,blocksize,tblocksize,x2,p2,x2t) = new(x,xt,n,p,blocksize,tblocksize,x2,p2,x2t)
end

# simple constructors for when n, p, and maybe blocksize are known and specified
# x must come from an actual BED file, so specify the path to the correct file
function BEDFile(
	T          :: Type,
	filename   :: ASCIIString, 
	tfilename  :: ASCIIString, 
	n          :: Int, 
	p          :: Int, 
	blocksize  :: Int, 
	tblocksize :: Int, 
	x2filename :: ASCIIString;
	pids       :: DenseVector{Int} = procs()
)
	x     = BEDFile(read_bedfile(filename, pids=pids),read_bedfile(tfilename, transpose = true, pids=pids),n,p,blocksize,tblocksize,SharedArray(T,n,0,pids=pids),0,SharedArray(T,0,n,pids=pids))
	x2    = readdlm(x2filename, T)
    x2_s  = SharedArray(T, size(x2), init = S -> localindexes(S) = zero(T), pids=pids) 
    copy!(x2_s, x2)
	p2    = size(x2,2)
	x.x2  = x2_s
	x.x2t = SharedArray(T, reverse(size(x2)), init = S -> localindexes(S) = zero(T), pids=pids) 
#	x.x2t = reshape(x2_s, reverse(size(x2_s))) 
	copy!(x.x2t, x2')
	x.p2  = p2
	return x
end

# for previous constructor, if type is not defined then default to Float64
BEDFile(filename::ASCIIString, tfilename::ASCIIString, n::Int, p::Int, blocksize::Int, tblocksize::Int, x2filename::ASCIIString; pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, tfilename, n, p, blocksize, tblocksize, x2filename, pids=pids)

function BEDFile(
	T          :: Type, 
	filename   :: ASCIIString, 
	tfilename  :: ASCIIString, 
	n          :: Int, 
	p          :: Int, 
	x2filename :: ASCIIString;
	pids       :: DenseVector{Int} = procs()
)
	x     = BEDFile(read_bedfile(filename, pids=pids),read_bedfile(tfilename, transpose=true, pids=pids),n,p,((n-1)>>>2)+1,((p-1)>>>2)+1,SharedArray(T,n,0,pids=pids),0, SharedArray(T,0,n,pids=pids), pids=pids)
	x2    = readdlm(x2filename)
    x2_s  = SharedArray(T, size(x2), init = S -> localindexes(S) = zero(T), pids=pids) 
    copy!(x2_s, x2)
	p2    = size(x2,2)
	x.x2  = x2_s
#	x.x2t = reshape(x2_s, reverse(size(x2_s))) 
	x.x2t = SharedArray(T, reverse(size(x2)), init = S -> localindexes(S) = zero(T), pids=pids) 
	copy!(x.x2t, x2')
	x.p2  = p2
	return x
end

# set default type for previous constructor to Float64
BEDFile(filename::ASCIIString, tfilename::ASCIIString, n::Int, p::Int, x2filename::ASCIIString; pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, tfilename, n, p, xtfilename, pids=pids)

# a more complicated constructor that attempts to infer n, p, and blocksize based on the BED filepath
# it assumes that the BED, FAM, and BIM files are all in the same directory
function BEDFile(T::Type, filename::ASCIIString, tfilename::ASCIIString; pids::DenseVector{Int} = procs())

	# find n from the corresponding FAM file 
	famfile = filename[1:(endof(filename)-3)] * "fam"
	n = count_cases(famfile)

	# find p from the corresponding BIM file
	bimfile = filename[1:(endof(filename)-3)] * "bim"
	p = count_predictors(bimfile)

	# blocksizes are easy to calculate
	blocksize  = ((n-1) >>> 2) + 1
	tblocksize = ((p-1) >>> 2) + 1

	# now load x, xt
	x   = read_bedfile(filename, pids=pids)
	xt  = read_bedfile(tfilename, transpose=true, pids=pids)
	x2  = SharedArray(T,n,0, init = S -> localindexes(S) = zero(T), pids=pids) 
	x2t = SharedArray(T,0,n, init = S -> localindexes(S) = zero(T), pids=pids) 
#	x2t = reshape(x2, reverse(size(x2))) 

	return BEDFile(x,xt,n,p,blocksize,tblocksize,x2,0,x2t)
end

# set default type for previous constructor to Float64
BEDFile(filename::ASCIIString, tfilename::ASCIIString; pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, tfilename, pids=pids)


# an extra constructor based on previous one 
# this one admits a third file path for the nongenetic covariates 
# it uncreatively creates a BEDFile using previous constructor with two file paths,
# and then fills the nongenetic covariates with the third file path 
function BEDFile(T::Type, filename::ASCIIString, tfilename::ASCIIString, x2filename::ASCIIString; header::Bool = false, pids::DenseVector{Int} = procs())

	x    = BEDFile(T, filename, tfilename, pids=pids)
	x2   = readdlm(x2filename, header=header)
	x.n   == size(x2,1) || throw(DimensionMismatch("Nongenetic covariates have more rows than genotype matrix"))
#    x2_s = SharedArray(T, size(x2), pids = pids)
#	copy!(x2_s, x2)
#	x.x2  = x2_s
	x.x2 = SharedArray(T, size(x2), pids = pids)
	copy!(x.x2, x2)
#	x.p2  = size(x2_s,2)
	x.p2 = size(x2,2)
#	x.x2t = reshape(x2_s, reverse(size(x2_s)))
	x.x2t = SharedArray(T, reverse(size(x2)), pids = pids)
	copy!(x.x2t, x2')
	return x 
end

# set default type for previous constructor to Float64
BEDFile(filename::ASCIIString, tfilename::ASCIIString, x2filename::ASCIIString; header::Bool = false, pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, tfilename, x2filename, header=header, pids=pids)


###########################
###  UTILITY FUNCTIONS ###
###########################

# COUNT PREDICTORS FROM NUMBER OF LINES IN BIM FILE
function count_predictors(f::ASCIIString)
	isequal(f[(endof(f)-3):endof(f)], ".bim") || throw(ArgumentError("Filename must point to a PLINK BIM file."))
	return countlines(f)
end

# COUNT CASES FROM NUMBER OF LINES IN FAM FILE
function count_cases(f::ASCIIString)
	isequal(f[(endof(f)-3):endof(f)], ".fam") || throw(ArgumentError("Filename must point to a PLINK FAM file."))
	return countlines(f)
end

# OBTAIN SIZE OF UNCOMPRESSED MATRIX
size(x::BEDFile) = (x.n, x.p + x.p2) 

function size(x::BEDFile, dim::Int)
	(dim == 1 || dim == 2) || throw(ArgumentError("Argument `dim` only accepts 1 or 2"))
	return ifelse(dim == 1, x.n, x.p + x.p2)
end

function size(x::BEDFile; submatrix::ASCIIString = "genotype")
	(isequal(submatrix, "genotype") || isequal(submatrix, "nongenetic")) || throw(ArgumentError("Argument `submatrix` only accepts `genotype` or `nongenetic`"))
	return ifelse(isequal(submatrix,"genotype"), (x.n, x.p), (x.n, x.p2))
end


# OBTAIN LENGTH OF UNCOMPRESSED MATRIX
length(x::BEDFile) = x.n*(x.p + x.p2)

# OBTAIN NUMBER OF DIMENSIONS OF UNCOMPRESSED MATRIX
ndims(x::BEDFile) = 2

# COPY A BEDFILE OBJECT
copy(x::BEDFile) = BEDFile(x.x, x.xt, x.n, x.p, x.blocksize, x.tblocksize, x.x2, x.p2, x.x2t)

# COMPARE DIFFERENT BEDFILE OBJECTS
==(x::BEDFile, y::BEDFile) = x.x   == y.x  &&
                             x.xt  == y.xt &&
                             x.n   == y.n  &&
                             x.p   == y.p  &&
                      x.blocksize  == y.blocksize &&
                     x.tblocksize  == y.tblocksize &&
					         x.x2  == y.x2 &&
					         x.p2  == y.p2 &&
							 x.x2t == y.x2t

isequal(x::BEDFile, y::BEDFile) = x == y 

function addx2!(x::BEDFile, x2::DenseArray{Float64,2}; pids::DenseVector{Int} = procs())
	(n,p2) = size(x2)
	n == x.n || throw(DimensionMismatch("x2 has $n rows but should have $(x.n) of them"))
	x.p2 = p2
	x.x2 = SharedArray(Float64, n, p2, init = S -> localindexes(S) = zero(Float64), pids=pids)
#	for j = 1:p2
#		for i = 1:x.n
#			@inbounds x.x2[i,j] = x2[i,j]
#		end
#	end
#	x.x2t = reshape(x.x2, reverse(size(x.x2)))
	copy!(x.x2,x2)
	x.x2t = SharedArray(Float64, p2, n, init = S -> localindexes(S) = zero(Float64), pids=pids)
	copy!(x.x2t, x2')
	return nothing
end


function addx2!(x::BEDFile, x2::DenseArray{Float32,2}; pids::DenseVector{Int} = procs())
	(n,p2) = size(x2)
	n == x.n || throw(DimensionMismatch("x2 has $n rows but should have $(x.n) of them"))
	x.p2 = p2
	x.x2 = SharedArray(Float32, n, p2, init = S -> localindexes(S) = zero(Float32), pids=pids)
#	for j = 1:p2
#		for i = 1:x.n
#			@inbounds x.x2[i,j] = x2[i,j]
#		end
#	end
#	x.x2t = reshape(x.x2, reverse(size(x.x2)))
	copy!(x.x2,x2)
	x.x2t = SharedArray(Float32, p2, n, init = S -> localindexes(S) = zero(Float32), pids=pids)
	copy!(x.x2t,x2')
	return nothing
end

function display(x::BEDFile)
	println("A BEDFile object with the following features:")
	println("\tnumber of cases        = $(x.n)")
	println("\tgenetic covariates     = $(x.p)")
	println("\tnongenetic covariates  = $(x.p2)")
	println("\tcovariate bits type    = $(typeof(x.x2))")
end

# READ PLINK BINARY GENOTYPE FILES
#
# This function reads a PLINK binary file (BED) and returns an array of Int8 numbers.
# It discards the first three bytes ("magic numbers") since they are not needed here.
# 
# Arguments:
# -- filename is the path to the BED file
#
# Output:
# -- A vector of Int8 numbers. For a genotype file with n cases and p SNPs,
#    there should be AT LEAST (n*p/4) numbers. The scaling factor of 4 comes from the
#    compression of four genotypes into each byte. But PLINK stores each column in blocks
#    of bytes instead of a continuous bitstream, which sometimes entails extra unused bits 
#    at the end of each block.
#
# coded by Kevin L. Keys (2015)
# klkeys@g.ucla.edu
function read_bedfile(filename::ASCIIString; transpose::Bool = false, pids::DenseVector{Int} = procs()) 

	# check that file is BED file
	contains(filename, ".bed") || throw(ArgumentError("Filename must point to a PLINK BED file."))

    # how many bytes do we have?
	# compute from IOStream to BED file
#	xstream = open(filename, "r")
#	nbytes = position(seekend(xstream))
#	close(xstream)
	nbytes = filesize(filename)

	# open file stream
	xstream = open(filename, "r")

	# check magic numbers and mode
#	isequal(read(xstream, Int8), MNUM1) || throw(error("Problem with first byte of magic number, is this a true BED file?"))
#	isequal(read(xstream, Int8), MNUM2) || throw(error("Problem with second byte of magic number, is this a true BED file?"))
#	(transpose && isequal(read(xstream, Int8), ONE8)) && throw(error("For transposed matrix, third byte of BED file must indicate individual-major format."))
	
	# now slurp file contents into SharedArray
#	x = SharedArray(abspath(filename), Int8, (nbytes,), pids=pids)

	# file seems to be a true BED file
	# will close filestream and slurp entire file into SharedArray
	close(xstream)
	x = SharedArray(abspath(filename), Int8, (nbytes-3,), 3, pids=pids)


	# return the genotypes
#	return x[4:end]
	return x
end


# SUBSET A COMPRESSED GENOTYPE MATRIX
#
# This subroutine will subset a stream of Int8 numbers representing a compressed genotype matrix.
# Argument X is vacuous; it simply ensures no ambiguity with current Array implementations
function subset_genotype_matrix(
	X         :: BEDFile, 
	x         :: DenseVector{Int8}, 
	rowidx    :: BitArray{1}, 
	colidx    :: BitArray{1}, 
	n         :: Int, 
	p         :: Int, 
	blocksize :: Int; 
	yn        :: Int = sum(rowidx), 
	yp        :: Int = sum(colidx), 
	yblock    :: Int = ((yn-1) >>> 2) + 1, 
	ytblock   :: Int = ((yp-1) >>> 2) + 1
)

	quiet = true 

	yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
	yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

	y = zeros(Int8, yp*yblock)
	(yn == 0 || yblock == 0) && return y

	l = 0
	# now loop over all columns in x 
	@inbounds for col = 1:p

		# only consider the current column of X if it is indexed
		if colidx[col]

			# count bytes in y
			l += 1

			# initialize a new block to fill
			new_block      = zero(Int8)
			num_genotypes  = 0
			current_row    = 0

			# start looping over cases
			@inbounds for row = 1:n

				# only consider the current row of X if it is indexed
				if rowidx[row]

					quiet || println("moving genotype for row = ", row, " and col = ", col)

					genotype = getindex(X,x,row,col,blocksize, interpret=false)

					# new_block stores the Int8 that we will eventually put in y
					# add new genotypes to it from the right
					# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
					new_block = new_block | (genotype << 2*num_genotypes) 

					quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

					# keep track of how many genotypes have been compressed so far 
					num_genotypes += 1

					quiet || println("num_genotypes is now ", num_genotypes)

					# make sure to track the number of cases that we have covered so far 
					current_row += 1
					quiet || println("current_row = ", current_row)

					# as soon as we pack the byte completely, then move to the next byte
					if num_genotypes == 4 && current_row < yn 
						y[l]          = new_block	# add new block to matrix y
						new_block     = zero(Int8)	# reset new_block
						num_genotypes = 0			# reset num_genotypes
						quiet || println("filled byte at l = ", l)

						# if not at last row, then increment the index for y
						# we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##						if sum(rowidx[1:min(row-1,n)]) !== yn
						if sum(rowidx[1:min(current_row-1,n)]) !== yn
##							quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
							quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
							l += 1		
							quiet || println("Incrementing l to l = ", l)
						end
					elseif current_row >= yn 
						# at this point, we haven't filled the byte
						# quit if we exceed the total number of cases
						# this will cause function to move to new genotype block
						quiet || println("Reached total number of rows, filling byte at l = ", l)
						y[l]          = new_block	# add new block to matrix y
						new_block     = zero(Int8)	# reset new_block
						num_genotypes = 0			# reset num_genotypes
						break
					end
				else
					# if current row is not indexed, then we merely add it to the counter
					# this not only ensures that its correspnding genotype is not compressed,
					# but it also ensures correct indexing for all of the rows in a column 
#					row += 1
				end # end if/else over current row 
			end # end loop over rows 
		end	# end if statement for current col 
	end	# end loop over cols 

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	return y

end


function subset_genotype_matrix(
	X         :: BEDFile, 
	x         :: DenseVector{Int8}, 
	rowidx    :: UnitRange{Int}, 
	colidx    :: BitArray{1}, 
	n         :: Int, 
	p         :: Int, 
	blocksize :: Int; 
	yn        :: Int = sum(rowidx), 
	yp        :: Int = sum(colidx), 
	yblock    :: Int = ((yn-1) >>> 2) + 1, 
	ytblock   :: Int = ((yp-1) >>> 2) + 1
)

	quiet = true 

	yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
	yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

	y = zeros(Int8, yp*yblock)
	(yn == 0 || yblock == 0) && return y

	l = 0
	# now loop over all columns in x 
	@inbounds for col = 1:p

		# only consider the current column of X if it is indexed
		if colidx[col]

			# count bytes in y
			l += 1

			# initialize a new block to fill
			new_block      = zero(Int8)
			num_genotypes  = 0
			current_row    = 0

			# start looping over cases
			@inbounds for row in rowidx 

				quiet || println("moving genotype for row = ", row, " and col = ", col)

				genotype = getindex(X,x,row,col,blocksize, interpret=false)

				# new_block stores the Int8 that we will eventually put in y
				# add new genotypes to it from the right
				# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
				new_block = new_block | (genotype << 2*num_genotypes) 

				quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

				# keep track of how many genotypes have been compressed so far 
				num_genotypes += 1

				quiet || println("num_genotypes is now ", num_genotypes)

				# make sure to track the number of cases that we have covered so far 
				current_row += 1
				quiet || println("current_row = ", current_row)

				# as soon as we pack the byte completely, then move to the next byte
				if num_genotypes == 4 && current_row < yn 
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					quiet || println("filled byte at l = ", l)

					# if not at last row, then increment the index for y
					# we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##						if sum(rowidx[1:min(row-1,n)]) !== yn
					if sum(rowidx[1:min(current_row-1,n)]) !== yn
##							quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
						quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
						l += 1		
						quiet || println("Incrementing l to l = ", l)
					end
				elseif current_row >= yn 
					# at this point, we haven't filled the byte
					# quit if we exceed the total number of cases
					# this will cause function to move to new genotype block
					quiet || println("Reached total number of rows, filling byte at l = ", l)
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					break
				end
			end # end loop over rows 
		end	# end if statement for current col 
	end	# end loop over cols 

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	return y

end



function subset_genotype_matrix(
	X         :: BEDFile, 
	x         :: DenseVector{Int8}, 
	rowidx    :: BitArray{1}, 
	colidx    :: UnitRange{Int}, 
	n         :: Int, 
	p         :: Int, 
	blocksize :: Int; 
	yn        :: Int = sum(rowidx), 
	yp        :: Int = sum(colidx), 
	yblock    :: Int = ((yn-1) >>> 2) + 1, 
	ytblock   :: Int = ((yp-1) >>> 2) + 1
)

	quiet = true 

	yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
	yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

	y = zeros(Int8, yp*yblock)
	(yn == 0 || yblock == 0) && return y

	l = 0
	# now loop over all columns in x 
	@inbounds for col in colidx 

		# count bytes in y
		l += 1

		# initialize a new block to fill
		new_block      = zero(Int8)
		num_genotypes  = 0
		current_row    = 0

		# start looping over cases
		@inbounds for row = 1:n

			# only consider the current row of X if it is indexed
			if rowidx[row]

				quiet || println("moving genotype for row = ", row, " and col = ", col)

				genotype = getindex(X,x,row,col,blocksize, interpret=false)

				# new_block stores the Int8 that we will eventually put in y
				# add new genotypes to it from the right
				# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
				new_block = new_block | (genotype << 2*num_genotypes) 

				quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

				# keep track of how many genotypes have been compressed so far 
				num_genotypes += 1

				quiet || println("num_genotypes is now ", num_genotypes)

				# make sure to track the number of cases that we have covered so far 
				current_row += 1
				quiet || println("current_row = ", current_row)

				# as soon as we pack the byte completely, then move to the next byte
				if num_genotypes == 4 && current_row < yn 
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					quiet || println("filled byte at l = ", l)

					# if not at last row, then increment the index for y
					# we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##						if sum(rowidx[1:min(row-1,n)]) !== yn
					if sum(rowidx[1:min(current_row-1,n)]) !== yn
##							quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
						quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
						l += 1		
						quiet || println("Incrementing l to l = ", l)
					end
				elseif current_row >= yn 
					# at this point, we haven't filled the byte
					# quit if we exceed the total number of cases
					# this will cause function to move to new genotype block
					quiet || println("Reached total number of rows, filling byte at l = ", l)
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					break
				end
			else
				# if current row is not indexed, then we merely add it to the counter
				# this not only ensures that its correspnding genotype is not compressed,
				# but it also ensures correct indexing for all of the rows in a column 
#					row += 1
			end # end if/else over current row 
		end # end loop over rows 
	end	# end loop over cols 

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	return y

end


function subset_genotype_matrix(
	X         :: BEDFile, 
	x         :: DenseVector{Int8}, 
	rowidx    :: UnitRange{Int}, 
	colidx    :: UnitRange{Int}, 
	n         :: Int, 
	p         :: Int, 
	blocksize :: Int; 
	yn        :: Int = sum(rowidx), 
	yp        :: Int = sum(colidx), 
	yblock    :: Int = ((yn-1) >>> 2) + 1, 
	ytblock   :: Int = ((yp-1) >>> 2) + 1
)

	quiet = true 

	yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
	yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

	y = zeros(Int8, yp*yblock)
	(yn == 0 || yblock == 0) && return y

	l = 0
	# now loop over all columns in x 
	@inbounds for col in colidx 

		# count bytes in y
		l += 1

		# initialize a new block to fill
		new_block      = zero(Int8)
		num_genotypes  = 0
		current_row    = 0

		# start looping over cases
		@inbounds for row in rowidx 

			# only consider the current row of X if it is indexed
			if rowidx[row]

				quiet || println("moving genotype for row = ", row, " and col = ", col)

				genotype = getindex(X,x,row,col,blocksize, interpret=false)

				# new_block stores the Int8 that we will eventually put in y
				# add new genotypes to it from the right
				# to do this, apply bitwise OR to new_block with genotype bitshifted left to correct position 
				new_block = new_block | (genotype << 2*num_genotypes) 

				quiet || println("Added ", genotype, " to new_block, which now equals ", new_block)

				# keep track of how many genotypes have been compressed so far 
				num_genotypes += 1

				quiet || println("num_genotypes is now ", num_genotypes)

				# make sure to track the number of cases that we have covered so far 
				current_row += 1
				quiet || println("current_row = ", current_row)

				# as soon as we pack the byte completely, then move to the next byte
				if num_genotypes == 4 && current_row < yn 
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					quiet || println("filled byte at l = ", l)

					# if not at last row, then increment the index for y
					# we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##						if sum(rowidx[1:min(row-1,n)]) !== yn
					if sum(rowidx[1:min(current_row-1,n)]) !== yn
##							quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
						quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
						l += 1		
						quiet || println("Incrementing l to l = ", l)
					end
				elseif current_row >= yn 
					# at this point, we haven't filled the byte
					# quit if we exceed the total number of cases
					# this will cause function to move to new genotype block
					quiet || println("Reached total number of rows, filling byte at l = ", l)
					y[l]          = new_block	# add new block to matrix y
					new_block     = zero(Int8)	# reset new_block
					num_genotypes = 0			# reset num_genotypes
					break
				end
			else
				# if current row is not indexed, then we merely add it to the counter
				# this not only ensures that its correspnding genotype is not compressed,
				# but it also ensures correct indexing for all of the rows in a column 
#					row += 1
			end # end if/else over current row 
		end # end loop over rows 
	end	# end loop over cols 

	# did we fill all of y?
	l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
	return y

end
