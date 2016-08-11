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
    filename   :: ASCIIString,
    tfilename  :: ASCIIString,
    n          :: Int,
    p          :: Int,
    blocksize  :: Int,
    tblocksize :: Int,
    x2filename :: ASCIIString,
    mfilename  :: ASCIIString,
    pfilename  :: ASCIIString;
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
    filename   :: ASCIIString,
    tfilename  :: ASCIIString, 
    n          :: Int, 
    p          :: Int, 
    blocksize  :: Int, 
    tblocksize :: Int, 
    x2filename :: ASCIIString,
    mfilename  :: ASCIIString,
    pfilename  :: ASCIIString;
    pids       :: DenseVector{Int} = procs(), 
    header     :: Bool = false
)
    BEDFile(Float64, filename, tfilename, n, p, blocksize, tblocksize, x2filename, mfilename, pfilename, pids=pids, header=header)
end

# constructor without blocksizes
function BEDFile(
    T          :: Type,
    filename   :: ASCIIString,
    tfilename  :: ASCIIString,
    n          :: Int,
    p          :: Int,
    x2filename :: ASCIIString,
    mfilename  :: ASCIIString,
    pfilename  :: ASCIIString;
    pids       :: DenseVector{Int} = procs(),
    header     :: Bool = false
)
    blocksize  = ( (n-1) >>> 2) + 1
    tblocksize = ( (p-1) >>> 2) + 1
    return BEDFile(T, filename, tfilename, n, p, blocksize, tblocksize, x2filename, mfilename, pfilename, pids=pids, header=header)
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: ASCIIString,
    tfilename  :: ASCIIString,
    n          :: Int, 
    p          :: Int, 
    x2filename :: ASCIIString,
    mfilename  :: ASCIIString,
    pfilename  :: ASCIIString;
    pids       :: DenseVector{Int} = procs(), 
    header     :: Bool = false
)
    BEDFile(Float64, filename, tfilename, n, p, xtfilename, mfilename, pfilename, pids=pids, header=header)
end


# constructor when only genotype data are available
# dummy means, precisions
function BEDFile(
    T         :: Type, 
    filename  :: ASCIIString, 
    tfilename :: ASCIIString; 
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
    filename  :: ASCIIString,
    tfilename :: ASCIIString; 
    pids      :: DenseVector{Int} = procs()
)
    BEDFile(Float64, filename, tfilename, pids=pids)
end


# constructor for when genotype, covariate information are available
# dummy means, precisions
function BEDFile(
    T          :: Type, 
    filename   :: ASCIIString, 
    tfilename  :: ASCIIString, 
    x2filename :: ASCIIString; 
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
    filename   :: ASCIIString, 
    tfilename  :: ASCIIString, 
    x2filename :: ASCIIString; 
    header     :: Bool = false, 
    pids       :: DenseVector{Int} = procs()
)
    BEDFile(Float64, filename, tfilename, x2filename, header=header, pids=pids)
end

# constructor to load all data from file
function BEDFile(
    T          :: Type, 
    filename   :: ASCIIString, 
    tfilename  :: ASCIIString, 
    x2filename :: ASCIIString,
    mfilename  :: ASCIIString,
    pfilename  :: ASCIIString; 
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
    filename   :: ASCIIString, 
    tfilename  :: ASCIIString, 
    x2filename :: ASCIIString,
    mfilename  :: ASCIIString,
    pfilename  :: ASCIIString; 
    pids       :: DenseVector{Int} = procs(),
    header     :: Bool = false, 
)
    BEDFile(Float64, filename, tfilename, x2filename, mfilename, pfilename, header=header, pids=pids)
end

# ambitious construtor that uses the location of the BED file and the covariates
# unlike other constructors, it will attempt to compute the correct means and precisions
function BEDFile(
    T          :: Type,
    filename   :: ASCIIString,
    x2filename :: ASCIIString;
    pids       :: DenseVector{Int} = procs(),
    header     :: Bool = false
)
    # find n from the corresponding FAM file
    famfile = filename[1:(endof(filename)-3)] * "fam"
    n       = countlines(famfile)

    # find p from the corresponding BIM file
    bimfile = filename[1:(endof(filename)-3)] * "bim"
    p       = countlines(bimfile)

    # use Desktop as a temporary directory for transpose
    # here we call our PLINK utility to transpose the file
    tmppath = expanduser("~/Desktop/tbed_$(myid()).bed")
    plinkpath = expanduser("~/.julia/v0.4/PLINK/utils/./plink_data")
    run(`$plinkpath $filename $p $n --transpose $tmppath`)

    # create a BEDFile object 
    x = BEDFile(T, filename, tmppath, x2filename, pids=pids, header=header)

    # calculate means and precisions
    mean!(x)
    prec!(x)

    # covariate mean/precision must be 0.0/1.0 
    x.means[end] = zero(T)
    x.precs[end] = one(T)

    # delete temporary files before returning
    rm(tmppath)

    return x
end

# default type for previous constructor is Float64
BEDFile(filename::ASCIIString, x2filename::ASCIIString; pids::DenseVector{Int} = procs(), header::Bool = false) = BEDFile(Float64, filename, x2filename, pids=pids, header=header) 

# another ambitious construtor that only uses the location of the BED file
# unlike previous constructors, the default covariate is a vector of ones
# also unlike other constructors, it will attempt to compute the correct means and precisions
function BEDFile(
    T        :: Type,
    filename :: ASCIIString;
    pids     :: DenseVector{Int} = procs()
)
    # make a temporary covariate file
    tmpcovar = expanduser("~/Desktop/x.txt")
    famfile  = filename[1:(endof(filename)-3)] * "fam"
    n        = countlines(famfile)
    writedlm(tmpcovar, ones(n))

    # create a BEDFile object 
    x = BEDFile(T, filename, tmpcovar, pids=pids, header=false)

    # delete temporary files before returning
    rm(tmpcovar)

    return x
end

# default type for previous constructor is Float64
BEDFile(filename::ASCIIString; pids::DenseVector{Int} = procs()) = BEDFile(Float64, filename, pids=pids) 

function display(x::BEDFile)
    println("A BEDFile object with the following features:")
    println("\tnumber of cases        = $(x.geno.n)")
    println("\tgenetic covariates     = $(x.geno.p)")
    println("\tnongenetic covariates  = $(x.covar.p)")
    println("\tcovariate array type   = $(eltype(x))")
end
