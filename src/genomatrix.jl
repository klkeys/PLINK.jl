# type definition
immutable GenoMatrix
    x          :: SharedVector{Int8}
    xt         :: SharedVector{Int8}
    n          :: Int
    p          :: Int
    blocksize  :: Int
    tblocksize :: Int
    snpids     :: Vector{UTF8String}

    GenoMatrix(x,xt,n,p,blocksize,tblocksize,snpids) = new(x,xt,n,p,blocksize,tblocksize,snpids)
end

# additional constructors, all of which will infer snpids field from BIM file
# consequently the snpids field never appers as an argument
# first constructor uses all remaining fields of GenoMatrix as arguments
function GenoMatrix(
    filename   :: ASCIIString,
    tfilename  :: ASCIIString,
    n          :: Int,
    p          :: Int,
    blocksize  :: Int,
    tblocksize :: Int;
    pids       :: DenseVector{Int} = procs(),
)
    # find SNP ids from the corresponding BIM file
    # first create filepath to BIM file
    bimfile = filename[1:(endof(filename)-3)] * "bim"

    # specify column element types of BIM file 
    eltypes = [Int, UTF8String, Int, Int, UTF8String, UTF8String]

    # load BIM
    df = readtable(bimfile, header = false, separator = '\t', eltypes = eltypes)

    # second column of BIM file contains the SNP ids
    snpids = convert(Vector{UTF8String}, df[:x2]) :: Vector{UTF8String} 
    
    x = GenoMatrix(
        read_bedfile(filename, pids=pids), 
        read_bedfile(tfilename, transpose = true, pids=pids),
        n, p, blocksize, tblocksize, snpids)
    
    return x
end

# another constructor for when blocksizes are not precomputed
function GenoMatrix(
    filename  :: ASCIIString,
    tfilename :: ASCIIString,
    n         :: Int,
    p         :: Int;
    pids      :: DenseVector{Int} = procs()
)
    blocksize  = ( (n-1) >>> 2) + 1
    tblocksize = ( (p-1) >>> 2) + 1
    return GenoMatrix(filename, tfilename, n, p, blocksize, tblocksize, pids=pids) :: PLINK.GenoMatrix
end


# constructor to load entirely from file
function GenoMatrix(
    filename  :: ASCIIString,
    tfilename :: ASCIIString;
    pids      :: DenseVector{Int} = procs()
)
    # find n from the corresponding FAM file
    famfile = filename[1:(endof(filename)-3)] * "fam"
    n       = countlines(famfile)

    # find p from the corresponding BIM file
    bimfile = filename[1:(endof(filename)-3)] * "bim"
    p       = countlines(bimfile)

    # blocksizes are easy to calculate
    blocksize  = ((n-1) >>> 2) + 1
    tblocksize = ((p-1) >>> 2) + 1

    # can now create GenoMatrix
    return GenoMatrix(filename, tfilename, n, p, blocksize, tblocksize, pids=pids) :: PLINK.GenoMatrix
end


# subroutines
Base.size(x::GenoMatrix) = (x.n, x.p)
Base.size(x::GenoMatrix, d::Int) = d == 1 ? x.n : x.p

Base.length(x::GenoMatrix) = x.n * x.p

Base.eltype(x::GenoMatrix) = Int8

Base.ndims(x::GenoMatrix) = 2

Base.copy(x::GenoMatrix) = GenoMatrix(x.x,x.xt,x.n,x.p,x.blocksize,x.tblocksize)

Base.linearindexing(x::Type{GenoMatrix}) = Base.LinearSlow() 


function ==(x::GenoMatrix, y::GenoMatrix)
    x.x  == y.x  &&
    x.xt == y.xt &&
    x.n  == y.n  &&
    x.p  == y.p  &&
    x.blocksize  == y.blocksize &&
    x.tblocksize == y.tblocksize
end

isequal(x::GenoMatrix, y::GenoMatrix) = x == y
    

# important matrix indexing! returns the Int8 
function getindex(
    X   :: GenoMatrix,
    x   :: DenseVector{Int8},
    row :: Int,
    col :: Int
)
    genotype_block = x[(col-1)*X.blocksize + ((row - 1) >>> 2) + 1]
    k = 2*((row-1) & 3)
    genotype = (genotype_block >>> k) & THREE8
    return genotype 
end

# internal routine to index the transposed matrix
# used for dott()
function getindex_t(
    X   :: GenoMatrix,
    x   :: DenseVector{Int8},
    row :: Int,
    col :: Int
)
    genotype_block = x[(col-1)*X.tblocksize + ((row - 1) >>> 2) + 1]
    k = 2*((row-1) & 3)
    genotype = (genotype_block >>> k) & THREE8
    return genotype 
end

# default indexing is to column-major matrix
# can use x.xt to index the transpose
getindex(x::GenoMatrix, row::Int, col::Int) = getindex(x, x.x, row, col)


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
function read_bedfile(
    filename  :: ASCIIString; 
    transpose :: Bool = false, 
    pids      :: DenseVector{Int} = procs()
)

    # check that file is BED file
    contains(filename, ".bed") || throw(ArgumentError("Filename must explicitly point to a PLINK \".bed\" file."))

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
    return x :: SharedVector{Int8}
end

function display(x::GenoMatrix) 
    println("A compressed GenoMatrix object with the following features:")
    println("\tnumber of cases        = $(x.n)")
    println("\tgenetic covariates     = $(x.p)")
end
