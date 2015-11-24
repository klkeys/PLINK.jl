"""

    BEDFile(x,xt,n,p,blocksize,tblocksize,x2,p2,x2t)

The `BEDFile` type encodes the compressed genotypes information housed in PLINK BED files.

Fields:

- `x` contains the compressed genotypes in a `SharedVector{Int8}`.
- `xt` contains the *transpose* of `x`, also stored as a `SharedVector{Int8}`.
- `n` is the number of cases.
- `p` is the number of SNPs.
- `blocksize` is the number of bytes per compressed column of genotype matrix.
- `tblocksize` is the number of bytes per compressed column of the *transposed* genotype matrix
- `x2` houses any nongenetic covariantes, if they exist.
- `p2` is the number of nongenetic covariates.
- `x2t = x2'`.

Note this BEDFile object, and the rest of this module for that matter, operate with the assumption
that the compressed matrix `x` is in column-major (SNP-major) format.
Row-major (case-major) format is used in the field `xt` but is not supported _per se_.
"""
type BEDFile
    x   :: SharedVector{Int8}
    xt  :: SharedVector{Int8}
    n   :: Int
    p   :: Int
    blocksize  :: Int
    tblocksize :: Int
    x2  :: SharedArray
    p2  :: Int
    x2t :: SharedArray

    BEDFile(x,xt,n,p,blocksize,tblocksize,x2,p2,x2t) = new(x,xt,n,p,blocksize,tblocksize,x2,p2,x2t)
end


"""

    BEDFile(T::Type, filename, tfilename, n, p, blocksize, tblocksize, x2filename [, pids=procs()])

Construct a `BEDFile` from filepaths `filename`, `tfilename`, and `x2filename` when `n`, `p`, `blocksize`, and `tblocksize` are known and specified.
If no argument is given for `T`, then it defaults to `Float64`.
"""
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
    copy!(x.x2t, x2')
    x.p2  = p2
    return x
end

# set default type for previous constructor to Float64
BEDFile(filename::ASCIIString, tfilename::ASCIIString, n::Int, p::Int, blocksize::Int, tblocksize::Int, x2filename::ASCIIString; pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, tfilename, n, p, blocksize, tblocksize, x2filename, pids=pids)



"""

    BEDFile(T::Type, filename, tfilename, n, p, x2filename [, pids=procs()])

Construct a `BEDFile` from filepaths `filename`, `tfilename`, and `x2filename` when `n` and `p` are known and specified.
The fields `blocksize` and `tblocksize` are inferred.
If no argument is given for `T`, then it defaults to `Float64`.
"""
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
    x.x2t = SharedArray(T, reverse(size(x2)), init = S -> localindexes(S) = zero(T), pids=pids)
    copy!(x.x2t, x2')
    x.p2  = p2
    return x
end

# set default type for previous constructor to Float64
BEDFile(filename::ASCIIString, tfilename::ASCIIString, n::Int, p::Int, x2filename::ASCIIString; pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, tfilename, n, p, xtfilename, pids=pids)

"""

    BEDFile(T::Type, filename, tfilename [, pids=procs()])

Construct a `BEDFile` from filepaths `filename`, `tfilename`, and `x2filename`.
The fields `n`, `p`, `blocksize` and `tblocksize` are inferred.
The PLINK FAM and BIM files are queried for calculating `n` and `p`.
They should lie in the same directory as the BED file specified by `filename`.
The fields for nongenetic covariates are initialized to zeroes.
If no argument is given for `T`, then it defaults to `Float64`.
"""
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

    return BEDFile(x,xt,n,p,blocksize,tblocksize,x2,0,x2t)
end

# set default type for previous constructor to Float64
BEDFile(filename::ASCIIString, tfilename::ASCIIString; pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, tfilename, pids=pids)


"""

    BEDFile(T::Type, filename, tfilename, x2filename [, pids=procs()])

Construct a `BEDFile` from filepaths `filename`, `tfilename`, and `x2filename`.
The fields `n`, `p`, `blocksize` and `tblocksize` are inferred.
The PLINK FAM and BIM files are queried for calculating `n` and `p`.
They should lie in the same directory as the BED file specified by `filename`.
The fields for nongenetic covariates are initialized to zeroes.
If no argument is given for `T`, then it defaults to `Float64`.
"""
function BEDFile(T::Type, filename::ASCIIString, tfilename::ASCIIString, x2filename::ASCIIString; header::Bool = false, pids::DenseVector{Int} = procs())

    x    = BEDFile(T, filename, tfilename, pids=pids)
    x2   = readdlm(x2filename, header=header)
    x.n   == size(x2,1) || throw(DimensionMismatch("Nongenetic covariates have more rows than genotype matrix"))
    x.x2 = SharedArray(T, size(x2), pids = pids)
    copy!(x.x2, x2)
    x.p2 = size(x2,2)
    x.x2t = SharedArray(T, reverse(size(x2)), pids = pids)
    copy!(x.x2t, x2')
    return x
end

# set default type for previous constructor to Float64
BEDFile(filename::ASCIIString, tfilename::ASCIIString, x2filename::ASCIIString; header::Bool = false, pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, tfilename, x2filename, header=header, pids=pids)


"Count the number of predictors `p` from a PLINK BIM file."
function count_predictors(f::ASCIIString)
    isequal(f[(endof(f)-3):endof(f)], ".bim") || throw(ArgumentError("Filename must point to a PLINK BIM file."))
    return countlines(f)
end

"Count the number of cases `n` from a PLINK FAM file."
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

"""

    addx2!(x::BEDFile, x2 [,pids=procs()])

Add a matrix of nongenetic covariates `x2` to a BEDFile `x`.
The optional argument `pids` controls the process IDs to which we distribute `x2`.
"""
function addx2!(x::BEDFile, x2::DenseMatrix{Float64}; pids::DenseVector{Int} = procs())
    (n,p2) = size(x2)
    n == x.n || throw(DimensionMismatch("x2 has $n rows but should have $(x.n) of them"))
    x.p2 = p2
    x.x2 = SharedArray(Float64, n, p2, init = S -> localindexes(S) = zero(Float64), pids=pids)
    copy!(x.x2,x2)
    x.x2t = SharedArray(Float64, p2, n, init = S -> localindexes(S) = zero(Float64), pids=pids)
    copy!(x.x2t, x2')
    return nothing
end


function addx2!(x::BEDFile, x2::DenseMatrix{Float32}; pids::DenseVector{Int} = procs())
    (n,p2) = size(x2)
    n == x.n || throw(DimensionMismatch("x2 has $n rows but should have $(x.n) of them"))
    x.p2 = p2
    x.x2 = SharedArray(Float32, n, p2, init = S -> localindexes(S) = zero(Float32), pids=pids)
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


"""

    read_bedfile(filename [, transpose=false, pids=procs]) -> SharedVector{Int8}

This function reads a PLINK binary file (BED) and returns a `SharedArray` of `Int8` numbers.
It discards the first three bytes ("magic numbers") since they are not needed here.

Arguments:

- `filename` is the path to the BED file

Optional Arguments:

- `transpose` indicates if the compressed matrix is transposed. Defaults to `false`.
- `pids` indicates the processes over which to distribute the output `SharedArray`. Defaults to `procs()` (all available processes).

Output:

- A vector of `Int8` numbers. For a BED file encoding `n` cases and `p` SNPs,
  there should be *at least* `(n*p/4)` numbers. The scaling factor of `4` comes from the
  compression of four genotypes into each byte. But PLINK stores each column in blocks
  of bytes instead of a continuous bitstream, which sometimes entails extra unused bits
  at the end of each block.
"""
function read_bedfile(filename::ASCIIString; transpose::Bool = false, pids::DenseVector{Int} = procs())

    # check that file is BED file
    contains(filename, ".bed") || throw(ArgumentError("Filename must point to a PLINK BED file."))

    # how many bytes do we have?
    nbytes = filesize(filename)

#    # open file stream
#    xstream = open(filename, "r")
#
#    # check magic numbers and mode
#   isequal(read(xstream, Int8), MNUM1) || throw(error("Problem with first byte of magic number, is this a true BED file?"))
#   isequal(read(xstream, Int8), MNUM2) || throw(error("Problem with second byte of magic number, is this a true BED file?"))
#   (transpose && isequal(read(xstream, Int8), ONE8)) && throw(error("For transposed matrix, third byte of BED file must indicate individual-major format."))
#
#    # now slurp file contents into SharedArray
#   x = SharedArray(abspath(filename), Int8, (nbytes,), pids=pids)

    # file seems to be a true BED file
    # will close filestream and slurp entire file into SharedArray
#    close(xstream)
    x = SharedArray(abspath(filename), Int8, (nbytes-3,), 3, pids=pids)

    # return the genotypes
#   return x[4:end]
    return x
end

"""
    subset_genotype_matrix(X, x, rowidx, colidx, n, p, blocksize) -> SharedVector{Int8}


This subroutine will subset a stream of `Int8` numbers representing a compressed genotype matrix.
Argument `X` is vacuous; it simply ensures no ambiguity with current `Array` implementations of `getindex`.

Arguments:

- `X` is a BEDFile object.
- `x` is either `X.x` or `X.xt`.
- `rowidx` indexes the rows to use in subsetting.
- `colidx` indexes the columns to subset.
- `n` is the number of cases.
- `p` is the number of predictors.
- `blocksize` is the number of bytes per column of the compressed matrix.

Optional Arguments:

- `pids` indicates the processes over which to distribute the output `SharedArray`. Defaults to `procs()` (all available processes).
- `yn` is the number of rows in the subset. When 'rowidx' is a `BitArray`, then `yn` defaults to `sum(rowidx)`, otherwise it defaults to `length(rowidx)`.
- `yp` is the number of columns in the subset. Similar to 'yn', `yp` defaults to `sum(colidx)` if `colidx` is a `BitArray` and `length(rowidx)` otherwise.
- `yblock` is the number of bytes per column of the subsetted matrix. Defaults to `((yn-1) >>> 2) + 1`.
- `ytblock` is the number of bytes per column of the subsetted matrix. Defaults to `((yp-1) >>> 2) + 1`.

Output:

- A `SharedArray` of type `Int8` that contains the subset of the BED file specified by `x`.
"""
function subset_genotype_matrix(
    X         :: BEDFile,
    x         :: DenseVector{Int8},
    rowidx    :: BitArray{1},
    colidx    :: BitArray{1},
    n         :: Int,
    p         :: Int,
    blocksize :: Int;
    pids      :: DenseVector{Int} = procs(),
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
                        y[l]          = new_block   # add new block to matrix y
                        new_block     = zero(Int8)  # reset new_block
                        num_genotypes = 0           # reset num_genotypes
                        quiet || println("filled byte at l = ", l)

                        # if not at last row, then increment the index for y
                        # we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##                      if sum(rowidx[1:min(row-1,n)]) !== yn
                        if sum(rowidx[1:min(current_row-1,n)]) !== yn
##                          quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
                            quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
                            l += 1
                            quiet || println("Incrementing l to l = ", l)
                        end
                    elseif current_row >= yn
                        # at this point, we haven't filled the byte
                        # quit if we exceed the total number of cases
                        # this will cause function to move to new genotype block
                        quiet || println("Reached total number of rows, filling byte at l = ", l)
                        y[l]          = new_block   # add new block to matrix y
                        new_block     = zero(Int8)  # reset new_block
                        num_genotypes = 0           # reset num_genotypes
                        break
                    end
                else
                    # if current row is not indexed, then we merely add it to the counter
                    # this not only ensures that its correspnding genotype is not compressed,
                    # but it also ensures correct indexing for all of the rows in a column
#                   row += 1
                end # end if/else over current row
            end # end loop over rows
        end # end if statement for current col
    end # end loop over cols

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
    pids      :: DenseVector{Int} = procs(),
    yn        :: Int = sum(rowidx),
    yp        :: Int = sum(colidx),
    yblock    :: Int = ((yn-1) >>> 2) + 1,
    ytblock   :: Int = ((yp-1) >>> 2) + 1
)

    quiet = true

    yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
    yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

    y = SharedArray(Int8, yp*yblock)
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
                    y[l]          = new_block   # add new block to matrix y
                    new_block     = zero(Int8)  # reset new_block
                    num_genotypes = 0           # reset num_genotypes
                    quiet || println("filled byte at l = ", l)

                    # if not at last row, then increment the index for y
                    # we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##                      if sum(rowidx[1:min(row-1,n)]) !== yn
                    if sum(rowidx[1:min(current_row-1,n)]) !== yn
##                          quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
                        quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
                        l += 1
                        quiet || println("Incrementing l to l = ", l)
                    end
                elseif current_row >= yn
                    # at this point, we haven't filled the byte
                    # quit if we exceed the total number of cases
                    # this will cause function to move to new genotype block
                    quiet || println("Reached total number of rows, filling byte at l = ", l)
                    y[l]          = new_block   # add new block to matrix y
                    new_block     = zero(Int8)  # reset new_block
                    num_genotypes = 0           # reset num_genotypes
                    break
                end
            end # end loop over rows
        end # end if statement for current col
    end # end loop over cols

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
    pids      :: DenseVector{Int} = procs(),
    yn        :: Int = sum(rowidx),
    yp        :: Int = sum(colidx),
    yblock    :: Int = ((yn-1) >>> 2) + 1,
    ytblock   :: Int = ((yp-1) >>> 2) + 1
)

    quiet = true

    yn <= n || throw(ArgumentError("rowidx indexes more rows than available in uncompressed matrix."))
    yp <= p || throw(ArgumentError("colidx indexes more columns than available in uncompressed matrix."))

    y = SharedArray(Int8, yp*yblock)
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
                    y[l]          = new_block   # add new block to matrix y
                    new_block     = zero(Int8)  # reset new_block
                    num_genotypes = 0           # reset num_genotypes
                    quiet || println("filled byte at l = ", l)

                    # if not at last row, then increment the index for y
                    # we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##                      if sum(rowidx[1:min(row-1,n)]) !== yn
                    if sum(rowidx[1:min(current_row-1,n)]) !== yn
##                          quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
                        quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
                        l += 1
                        quiet || println("Incrementing l to l = ", l)
                    end
                elseif current_row >= yn
                    # at this point, we haven't filled the byte
                    # quit if we exceed the total number of cases
                    # this will cause function to move to new genotype block
                    quiet || println("Reached total number of rows, filling byte at l = ", l)
                    y[l]          = new_block   # add new block to matrix y
                    new_block     = zero(Int8)  # reset new_block
                    num_genotypes = 0           # reset num_genotypes
                    break
                end
            else
                # if current row is not indexed, then we merely add it to the counter
                # this not only ensures that its correspnding genotype is not compressed,
                # but it also ensures correct indexing for all of the rows in a column
#                   row += 1
            end # end if/else over current row
        end # end loop over rows
    end # end loop over cols

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

    y = SharedArray(Int8, yp*yblock)
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
                    y[l]          = new_block   # add new block to matrix y
                    new_block     = zero(Int8)  # reset new_block
                    num_genotypes = 0           # reset num_genotypes
                    quiet || println("filled byte at l = ", l)

                    # if not at last row, then increment the index for y
                    # we skip incrementing l at the last row to avoid double-incrementing l at start of new predictor
##                      if sum(rowidx[1:min(row-1,n)]) !== yn
                    if sum(rowidx[1:min(current_row-1,n)]) !== yn
##                          quiet || println("currently at ", sum(rowidx[1:min(row-1,n)]), " rows of ", yn, " total.")
                        quiet || println("currently at ", sum(rowidx[1:min(current_row-1,n)]), " rows of ", yn, " total.")
                        l += 1
                        quiet || println("Incrementing l to l = ", l)
                    end
                elseif current_row >= yn
                    # at this point, we haven't filled the byte
                    # quit if we exceed the total number of cases
                    # this will cause function to move to new genotype block
                    quiet || println("Reached total number of rows, filling byte at l = ", l)
                    y[l]          = new_block   # add new block to matrix y
                    new_block     = zero(Int8)  # reset new_block
                    num_genotypes = 0           # reset num_genotypes
                    break
                end
            else
                # if current row is not indexed, then we merely add it to the counter
                # this not only ensures that its correspnding genotype is not compressed,
                # but it also ensures correct indexing for all of the rows in a column
#                   row += 1
            end # end if/else over current row
        end # end loop over rows
    end # end loop over cols

    # did we fill all of y?
    l == length(y) || warn("subsetted matrix x has $(length(y)) indices but we filled $l of them")
    return y

end
