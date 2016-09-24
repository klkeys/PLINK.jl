### subroutines to read data

"""
    read_data(xpath) -> x::BEDFile, y::SharedVector

Read the trio of binary PLINK files (BED, BIM, FAM) from `xpath`. The response `y` is loaded from the rightmost column of the FAM file.
Unlike the `BEDFile` constructors, the variable `xpath` follows the PLINK convention in that it should point to the trio of files.
For example,

    read_data("mydata")

would read the trio `mydata.bed`, `mydata.bim`, and `mydata.fam` in the current working directory.
"""
function read_plink_data(
    T      :: Type, 
    xpath  :: String; 
    pids   :: DenseVector{Int} = procs(),
    header :: Bool = false,
    delim  :: Char = ' '
)

    # load genotype data
    x = BEDFile(T, filepath * ".bed", pids=pids)

    # read the FAM file
    famfile = filepath * ".fam"
    Y = readdlm(famfile, delim, header=header)

    # check that the FAM file has six columns
    p = size(Y,2)
    p == 6 || throw(DimensionMismatch("FAM file does not have six columns, is it formatted correctly?"))

    # we cannot know for certain that Y is loaded as floating point
    # to be safe, explicitly convert the phenotype column to the correct type
    ytemp = convert(Vector{T}, Y[:,end]) 

    # in FAM file, the phenotype is the rightmost column 
    # initialize a SharedVector and fill it with the phenotype
    y = SharedArray(T, (x.geno.n,), pids=pids)
    copy!(y, ytemp) 

    return x, y
end

# default for previous function is Float64
read_plink_data(filepath::String; pids::DenseVector{Int} = procs(), header::Bool = false, delim::Char = ' ') = read_plink_data(Float64, filepath, pids=pids, header=header, delim=delim)


"""
    read_plink_data(xpath, ypath [,isbin=false, header=false, delim=' ']) -> x::BEDFile, y::SharedVector

Read the trio of binary PLINK files (BED, BIM, FAM) based on the file path `xpath`.
By default, the response `y` is loaded from a delimited text file located at `ypath`.
Set `isbin=true` to load from a _binary_ file.
"""
function read_plink_data(
    T      :: Type, 
    xpath  :: String, 
    ypath  :: String; 
    pids   :: DenseVector{Int} = procs(),
    isbin  :: Bool = false,
    header :: Bool = false,
    delim  :: Char = ' '
)

    # load genotype data
    x = BEDFile(T, filepath * ".bed", pids=pids)

    # load phenotype from file
    # binary files -> make SharedVector directly
    # delimited text -> must wrangle into SharedVector
    if isbin
        y = SharedArray(abspath(ypath), T, (x.geno.n,), pids=pids)
    else
        ytemp = readdlm(ypath, delim, header=header)
        size(ytemp,2) == 1 || throw(DimensionMismatch("File at ypath must have exactly one column"))
        y = SharedArray(T, (x.geno.n,), pids=pids)
        copy!(y, ytemp)
    end

    return x, y
end

# default for previous function is Float64
read_plink_data(xpath::String, ypath::String, pids::DenseVector{Int} = procs(), isbin::Bool = false, header::Bool = false, delim::Char = ' ') = read_plink_data(Float64, xpath, ypath, pids=pids, isbin=isbin, header=header, delim=delim)


"""
    read_plink_data(xpath, covpath, ypath [,isbin=false, header=false, delim=' ']) -> x::BEDFile, y::SharedVector

Read the trio of binary PLINK files (BED, BIM, FAM) based on the file path `xpath`, plus covariates from the file path `covpath`. 
By default, the response `y` is loaded from a delimited text file located at `ypath`.
Set `isbin=true` to load from a _binary_ file.
"""
function read_plink_data(
    T         :: Type, 
    xpath     :: String, 
    covpath   :: String, 
    ypath     :: String;
    pids      :: DenseVector{Int} = procs(),
    covheader :: Bool = false,
    covdelim  :: Char = ' ',
    isbin     :: Bool = false,
    yheader   :: Bool = false,
    ydelim    :: Char = ' ',
)

    # load genotype data
    x = BEDFile(T, filepath * ".bed", covpath, pids=pids, header=covheader)

    # load phenotype from file
    # binary files -> make SharedVector directly
    # delimited text -> must wrangle into SharedVector
    if isbin
        y = SharedArray(abspath(ypath), T, (x.geno.n,), pids=pids)
    else
        ytemp = readdlm(ypath, ydelim, header=yheader)
        size(ytemp,2) == 1 || throw(DimensionMismatch("File at ypath must have exactly one column"))
        y = SharedArray(T, (x.geno.n,), pids=pids)
        copy!(y, ytemp)
    end

    return x, y
end

# default for previous function is Float64
read_plink_data(xpath::String, covpath::String, ypath::String, pids::DenseVector{Int} = procs(), covheader::Bool = false, covdelim::Char = ' ', isbin::Bool = false, yheader::Bool = false, ydelim::Char = ' ' ) = read_plink_data(Float64, xpath, covpath, ypath, pids=pids, covheader=covheader, covdelim=covdevlim, isbin=isbin, yheader=yheader, ydelim=ydelim)
