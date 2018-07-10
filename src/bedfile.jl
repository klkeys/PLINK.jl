immutable BEDFile{T <: Float} <: AbstractMatrix{T}
    geno  :: GenoMatrix
    covar :: CovariateMatrix{T}
    means :: SharedVector{T}
    precs :: SharedVector{T}
    mask  :: Vector{Int}
end

BEDFile{T <: Float}(geno::GenoMatrix, covar::CovariateMatrix, means::SharedVector{T}, precs::SharedVector{T}, mask::Vector{Int}) = BEDFile{eltype(covar)}(geno, covar, means, precs, mask)

# subroutines for AbstractMatrix type
Base.size(x::BEDFile) = (x.geno.n, x.geno.p + x.covar.p)
Base.size(x::BEDFile, d::Int) = d == 1? x.geno.n : size(x.geno, 2) + size(x.covar, 2)

Base.eltype(x::BEDFile) = eltype(x.covar)

Base.ndims(x::BEDFile) = 2

Base.copy(x::BEDFile) = BEDFile(x.geno, x.covar, x.means, x.precs, x.mask)

function ==(x::BEDFile, y::BEDFile)
     x.geno == y.geno  &&
    x.covar == y.covar &&
    x.means == y.means &&
    x.precs == y.precs &&
    x.ids   == y.ids
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

# matrix manipulations
# what to do about transpose in CovariateMatrix?
function setindex!{T <: Float}(x::BEDFile{T}, v, i::Int)
    @assert i > x.p "Cannot index changes into genotype array of BEDFile object since it is read-only"
    setindex!(x.x, v, i-x.p)
end

typeof(x::BEDFile) = "$(x.n)x$(size(x,2)) BEDFile{$(eltype(x.x))} with $(x.p2) nongenetic covariates"
summary(x::BEDFile) = typeof(x)

# get names of all predictors
prednames(x::BEDFile) = [x.geno.snpids; x.covar.h]


### BEDFile constructors

# construct (almost) everything from file
function BEDFile(
    T          :: Type,
    filename   :: String,
    tfilename  :: String,
    n          :: Int,
    p          :: Int,
    blocksize  :: Int,
    tblocksize :: Int,
    x2filename :: String,
    mfilename  :: String,
    pfilename  :: String;
    pids       :: Vector{Int} = procs(),
    header     :: Bool = false
)
    x = GenoMatrix(filename, tfilename, n, p, blocksize, tblocksize, pids=pids) :: GenoMatrix{T}
    y = CovariateMatrix(T,x2filename, pids=pids, header=header) :: CovariateMatrix{T}
    m = SharedArray{T}(abspath(mfilename), (x.p + y.p,), pids=pids) :: SharedVector{T}
    d = SharedArray{T}(abspath(pfilename), (x.p + y.p,), pids=pids) :: SharedVector{T}
    return BEDFile{T}(x,y,m,d,ones(Int,x.n))
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: String,
    tfilename  :: String,
    n          :: Int,
    p          :: Int,
    blocksize  :: Int,
    tblocksize :: Int,
    x2filename :: String,
    mfilename  :: String,
    pfilename  :: String;
    pids       :: Vector{Int} = procs(),
    header     :: Bool = false
)
    BEDFile(Float64, filename, tfilename, n, p, blocksize, tblocksize, x2filename, mfilename, pfilename, pids=pids, header=header) :: BEDFile{Float64}
end

# constructor without blocksizes
function BEDFile(
    T          :: Type,
    filename   :: String,
    tfilename  :: String,
    n          :: Int,
    p          :: Int,
    x2filename :: String,
    mfilename  :: String,
    pfilename  :: String;
    pids       :: Vector{Int} = procs(),
    header     :: Bool = false
)
    blocksize  = ( (n-1) >>> 2) + 1
    tblocksize = ( (p-1) >>> 2) + 1
    return BEDFile(T, filename, tfilename, n, p, blocksize, tblocksize, x2filename, mfilename, pfilename, pids=pids, header=header) :: BEDFile{T}
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: String,
    tfilename  :: String,
    n          :: Int,
    p          :: Int,
    x2filename :: String,
    mfilename  :: String,
    pfilename  :: String;
    pids       :: Vector{Int} = procs(),
    header     :: Bool = false
)
    BEDFile(Float64, filename, tfilename, n, p, xtfilename, mfilename, pfilename, pids=pids, header=header) :: BEDFile{Float64}
end


# constructor for when genotype, covariate information are available
# computes means, precisions
function BEDFile(
    T          :: Type,
    filename   :: String,
    tfilename  :: String,
    x2filename :: String;
    pids       :: Vector{Int} = procs(),
    header     :: Bool = false,
)
    # making matrices is easy
    x = GenoMatrix(filename, tfilename, pids=pids)
    y = CovariateMatrix(T, x2filename, pids=pids, header=header)

    # but ensure same number of rows!
    @assert x.n == size(y,1) "Nongenetic covariates and genotype matrix must have equal number of rows"

    # make dummy means, precisions
    # default yields no standardization (zero mean, identity precision)
    m = SharedArray{T}((x.p + y.p,), init = S -> localindexes(S) = zero(T), pids=pids) :: SharedVector{T}
    d = SharedArray{T}((x.p + y.p,), init = S -> localindexes(S) = one(T),  pids=pids) :: SharedVector{T}

    # construct BEDFile
    z = BEDFile{T}(x,y,m,d,ones(Int,x.n))

    # compute means, precisions
    mean!(z);
    prec!(z);

    # loop through covariate matrix
    # if we have a column with everything equal to the first element of column,
    # then change the mean/prec for that covariate to 0/1
    for i = 1:z.covar.p
        if all(view(z.covar.x, :, i) .== z.covar.x[1,i])
            z.means[z.geno.p + i] = zero(T)
            z.precs[z.geno.p + i] = one(T)
        end
    end

    return z :: BEDFile{T}
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: String,
    tfilename  :: String,
    x2filename :: String;
    header     :: Bool = false,
    pids       :: Vector{Int} = procs()
)
    BEDFile(Float64, filename, tfilename, x2filename, header=header, pids=pids) :: BEDFile{Float64}
end

# constructor to load all data from file
function BEDFile(
    T          :: Type,
    filename   :: String,
    tfilename  :: String,
    x2filename :: String,
    mfilename  :: String,
    pfilename  :: String;
    pids       :: Vector{Int} = procs(),
    header     :: Bool = false,
)
    # making matrices is easy
    x = GenoMatrix(filename, tfilename, pids=pids)
    y = CovariateMatrix(T, x2filename, pids=pids, header=header)

    # but ensure same number of rows!
    @assert x.n == size(y,1) "Nongenetic covariates and genotype matrix must have equal number of rows"

    # load means, precisions
    p = x.p + y.p
    m = SharedArray{T}(abspath(mfilename), (p,), pids=pids) :: SharedVector{T}
    d = SharedArray{T}(abspath(pfilename), (p,), pids=pids) :: SharedVector{T}

    return BEDFile{T}(x,y,m,d,ones(Int,x.n))
end

# set default type for previous constructor to Float64
function BEDFile(
    filename   :: String,
    tfilename  :: String,
    x2filename :: String,
    mfilename  :: String,
    pfilename  :: String;
    pids       :: Vector{Int} = procs(),
    header     :: Bool = false,
)
    BEDFile(Float64, filename, tfilename, x2filename, mfilename, pfilename, header=header, pids=pids) :: BEDFile{Float64}
end

# ambitious construtor that uses the location of the BED file and the covariates
# computes means, precisions
function BEDFile(
    T          :: Type,
    filename   :: String,
    x2filename :: String;
    pids       :: Vector{Int} = procs(),
    header     :: Bool = false
)
    # find n from the corresponding FAM file
    famfile = filename[1:(endof(filename)-3)] * "fam"
    n       = countlines(famfile)

    # find p from the corresponding BIM file
    bimfile = filename[1:(endof(filename)-3)] * "bim"
    p       = countlines(bimfile)

    # use Julia temporary directory for transpose
    # here we call our PLINK utility to transpose the file
    tmppath = tempdir() * "/tbed_$(myid()).bed"
    vernum  = "v$(VERSION.major).$(VERSION.minor)"
    plinkpath = expanduser("~/.julia/$vernum/PLINK/utils/./plink_data")
    run(`$plinkpath $filename $p $n --transpose $tmppath`)

    # create a BEDFile object
    x = BEDFile(T, filename, tmppath, x2filename, pids=pids, header=header) :: BEDFile{T}

    # calculate means and precisions
    mean!(x)
    prec!(x)

    # loop through covariate matrix
    # if we have a column with everything equal to the first element of column,
    # then change the mean/prec for that covariate to 0/1
    for i = 1:x.covar.p
        if all(view(x.covar.x, :, i) .== x.covar.x[1,i])
            x.means[x.geno.p + i] = zero(T)
            x.precs[x.geno.p + i] = one(T)
        end
    end

    # delete temporary files before returning
    rm(tmppath)

    return x :: BEDFile{T}
end

# default type for previous constructor is Float64
BEDFile(filename::String, x2filename::String; pids::Vector{Int} = procs(), header::Bool = false) = BEDFile(Float64, filename, x2filename, pids=pids, header=header) :: BEDFile{Float64}

# another ambitious construtor that only uses the location of the BED file
# unlike previous constructors, the default covariate is a vector of ones
# also unlike other constructors, it will attempt to compute the correct means and precisions
function BEDFile(
    T        :: Type,
    filename :: String;
    pids     :: Vector{Int} = procs()
)
    # make a temporary covariate file
    tmpcovar = tempdir() * "/x_$(myid()).txt"
    famfile  = filename[1:(endof(filename)-3)] * "fam"
    n        = countlines(famfile)
    writedlm(tmpcovar, ones(n))

    # create a BEDFile object
    x = BEDFile(T, filename, tmpcovar, pids=pids, header=false) :: BEDFile{T}

    # delete temporary files before returning
    rm(tmpcovar)

    return x :: BEDFile{T}
end

# default type for previous constructor is Float64
BEDFile(filename::String; pids::Vector{Int} = procs()) = BEDFile(Float64, filename, pids=pids) :: BEDFile{Float64}

function display(x::BEDFile)
    println("A BEDFile object with the following features:")
    println("\tnumber of cases        = $(x.geno.n)")
    println("\tgenetic covariates     = $(x.geno.p)")
    println("\tnongenetic covariates  = $(x.covar.p)")
    println("\tcovariate array type   = $(eltype(x))")
end
