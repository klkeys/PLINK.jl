immutable BEDFile{T <: Float, V <: SharedMatrix}
    geno  :: GenoMatrix
    covar :: CovariateMatrix{T, V}
    means :: SharedVector{T}
    precs :: SharedVector{T}
end

BEDFile(geno::GenoMatrix, covar::CovariateMatrix, means::SharedVector, precs::SharedVector) = BEDFile{eltype(covar)}(geno, covar, means, precs)

# subroutines
Base.size(x::BEDFile) = (x.geno.n, x.geno.p + x.covar.p)
Base.size(x::BEDFile, d::Int) = d == 1? x.geno.n : size(x.geno, d) + size(x.covar, 2) 

Base.eltype(x::BEDFile) = eltype(x.covar)

Base.ndims(x::BEDFile) = 2

Base.copy(x::BEDFile) = BEDFile(x.geno, x.covar, x.means, x.precs)

function ==(x::BEDFile, y::BEDFile)
     x.geno == y.geno  && 
    x.covar == y.covar && 
    x.means == y.means && 
    x.precs == y.precs
end

Base.isequal(x::BEDFile, y::BEDFile) = x == y

Base.procs(x::BEDFile) = procs(x.geno.x)

@inline function int2geno{T <: Float}(
    x :: BEDFile{T},
    i :: Int8
)
    convert(T, genofloat[i + ONE8])
end


# matrix indexing
function getindex{T <: Float}(
    x   :: BEDFile{T},
    row :: Int, 
    col :: Int
)
    col > x.geno.p && return getindex(x.covar, row, col-x.geno.p)
    return int2geno(x, x.geno[row, col])
end

function BEDFile(
    T          :: Type,
    filename   :: AbstractString,
    tfilename  :: AbstractString,
    n          :: Int,
    p          :: Int,
    blocksize  :: Int,
    tblocksize :: Int,
    x2filename :: AbstractString,
    mfilename  :: AbstractString,
    pfilename  :: AbstractString;
    pids       :: DenseVector{Int} = procs(),
    header     :: Bool = false
)
    x = GenoMatrix(filename, tfilename, n, p, blocksize, tblocksize, pids=pids)
    y = CovariateMatrix(T,x2filename, pids=pids, header=header)
    m = SharedArray(abspath(mfilename), T, (x.p + y.p,), pids=pids)
    d = SharedArray(abspath(pfilename), T, (x.p + y.p,), pids=pids)
    return BEDFile(x,y,m,d)
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: AbstractString,
    tfilename  :: AbstractString, 
    n          :: Int, 
    p          :: Int, 
    blocksize  :: Int, 
    tblocksize :: Int, 
    x2filename :: AbstractString,
    mfilename  :: AbstractString,
    pfilename  :: AbstractString;
    pids       :: DenseVector{Int} = procs(), 
    header     :: Bool = false
)
    BEDFile(Float64, filename, tfilename, n, p, blocksize, tblocksize, x2filename, mfilename, pfilename, pids=pids, header=header)
end

# constructor without blocksizes
function BEDFile(
    T          :: Type,
    filename   :: AbstractString,
    tfilename  :: AbstractString,
    n          :: Int,
    p          :: Int,
    x2filename :: AbstractString,
    mfilename  :: AbstractString,
    pfilename  :: AbstractString;
    pids       :: DenseVector{Int} = procs(),
    header     :: Bool = false
)
    blocksize  = ( (n-1) >>> 2) + 1
    tblocksize = ( (p-1) >>> 2) + 1
    return BEDFile(T, filename, tfilename, n, p, blocksize, tblocksize, x2filename, mfilename, pfilename, pids=pids, header=header)
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: AbstractString,
    tfilename  :: AbstractString,
    n          :: Int, 
    p          :: Int, 
    x2filename :: AbstractString,
    mfilename  :: AbstractString,
    pfilename  :: AbstractString;
    pids       :: DenseVector{Int} = procs(), 
    header     :: Bool = false
)
    BEDFile(Float64, filename, tfilename, n, p, xtfilename, mfilename, pfilename, pids=pids, header=header)
end


# constructor when only genotype data are available
# dummy means, precisions
function BEDFile(
    T         :: Type, 
    filename  :: AbstractString, 
    tfilename :: AbstractString; 
    pids      :: DenseVector{Int} = procs()
)

    # can easily create GenoMatrix
    x = GenoMatrix(filename, tfilename, pids=pids)

    # now make dummy CovariateMatrix 
    x2  = SharedArray(T, x.n, 1, init = S -> localindexes(S) = zero(T), pids=pids)
    x2t = SharedArray(T, 1, x.n, init = S -> localindexes(S) = zero(T), pids=pids)
    y   = CovariateMatrix(x2,1,x2t)

    # make dummy means, precisions
    # default yields no standardization (zero mean, identity precision)
    p = x.p + y.p
    m = SharedArray(T, (p,), init = S -> localindexes(S) = zero(T), pids=pids) 
    d = SharedArray(T, (p,), init = S -> localindexes(S) = one(T),  pids=pids) 

    return BEDFile(x,y,m,d)
end

# set default type for previous constructor to Float64
function BEDFile(
    filename  :: AbstractString,
    tfilename :: AbstractString; 
    pids      :: DenseVector{Int} = procs()
)
    BEDFile(Float64, filename, tfilename, pids=pids)
end


# constructor for when genotype, covariate information are available
# dummy means, precisions
function BEDFile(
    T          :: Type, 
    filename   :: AbstractString, 
    tfilename  :: AbstractString, 
    x2filename :: AbstractString; 
    pids       :: DenseVector{Int} = procs(),
    header     :: Bool = false, 
)
    # making matrices is easy
    x = GenoMatrix(filename, tfilename, pids=pids)
    y = CovariateMatrix(T, x2filename, pids=pids, header=header)

    # but ensure same number of rows!
    x.n == size(y,1) || throw(DimensionMismatch("Nongenetic covariates and genotype matrix must have equal number of rows"))

    # make dummy means, precisions
    # default yields no standardization (zero mean, identity precision)
    m = SharedArray(T, (x.p + y.p,), init = S -> localindexes(S) = zero(T), pids=pids) 
    d = SharedArray(T, (x.p + y.p,), init = S -> localindexes(S) = one(T),  pids=pids) 

    return BEDFile(x,y,m,d) 
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: AbstractString, 
    tfilename  :: AbstractString, 
    x2filename :: AbstractString; 
    header     :: Bool = false, 
    pids       :: DenseVector{Int} = procs()
)
    BEDFile(Float64, filename, tfilename, x2filename, header=header, pids=pids)
end

# constructor to load all data from file
function BEDFile(
    T          :: Type, 
    filename   :: AbstractString, 
    tfilename  :: AbstractString, 
    x2filename :: AbstractString,
    mfilename  :: AbstractString,
    pfilename  :: AbstractString; 
    pids       :: DenseVector{Int} = procs(),
    header     :: Bool = false, 
)
    # making matrices is easy
    x = GenoMatrix(filename, tfilename, pids=pids)
    y = CovariateMatrix(T, x2filename, pids=pids, header=header)

    # but ensure same number of rows!
    x.n == size(y,1) || throw(DimensionMismatch("Nongenetic covariates and genotype matrix must have equal number of rows"))

    # load means, precisions
    p = x.p + y.p
    m = SharedArray(abspath(mfilename), T, (p,), pids=pids)
    d = SharedArray(abspath(pfilename), T, (p,), pids=pids)

    return BEDFile(x,y,m,d) 
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: AbstractString, 
    tfilename  :: AbstractString, 
    x2filename :: AbstractString,
    mfilename  :: AbstractString,
    pfilename  :: AbstractString; 
    pids       :: DenseVector{Int} = procs(),
    header     :: Bool = false, 
)
    BEDFile(Float64, filename, tfilename, x2filename, mfilename, pfilename, header=header, pids=pids)
end


function display(x::BEDFile)
    println("A BEDFile object with the following features:")
    println("\tnumber of cases        = $(x.geno.n)")
    println("\tgenetic covariates     = $(x.geno.p)")
    println("\tnongenetic covariates  = $(x.covar.p)")
    println("\tcovariate array type   = $(eltype(x))")
end
