# container for nongenetic covariates
immutable CovariateMatrix{T <: Float} <: AbstractArray{T, 2}
    x  :: SharedMatrix{T} 
    p  :: Int
    xt :: SharedMatrix{T} 
    h  :: Vector{String} # header 

    CovariateMatrix(x::SharedMatrix{T}, p::Int, xt::SharedMatrix{T}, h::Vector{String}) = new(x,p,xt,h)
end

### 22 Sep 2016: not needed in Julia v0.5?
#function CovariateMatrix{T <: Float}(
#    x  :: SharedMatrix{T},
#    p  :: Int,
#    xt :: SharedMatrix{T},
#    h  :: Vector{String}
#)
#    CovariateMatrix{eltype(x)}(x, p, xt, h) :: CovariateMatrix{eltype(x)}
#end

function CovariateMatrix(
    T        :: Type, 
    filename :: String;
    pids     :: Vector{Int} = procs(),
    header   :: Bool = false
)
    # first load data
    if header
        x_temp, h = readdlm(filename, T, header=true) :: Matrix{T}, Matrix{AbstractString}
        h = convert(Vector{String}, h) :: Vector{String}
    else
        x_temp = readdlm(filename, T, header=false) :: Matrix{T}
        h = ["V" * "$i" for i = 1:size(x_temp,2)] :: Vector{String}
    end

    # make SharedArray container for x_temp
    x = SharedArray(T, size(x_temp), pids=pids) :: SharedMatrix{eltype(x_temp)}
    copy!(x, x_temp)

    # do same for x'
    xt = SharedArray(T, reverse(size(x_temp)), pids=pids) :: SharedMatrix{eltype(x_temp)}
    transpose!(xt, x_temp)

    # save p for convenience
    p = size(x,2)

    return CovariateMatrix{eltype(x)}(x, p, xt, h) :: CovariateMatrix{eltype(x)}
end

# default to Float64 constructor
function CovariateMatrix(
    filename :: String;
    pids     :: Vector{Int} = procs(),
    header   :: Bool = false
)
    CovariateMatrix(Float64, filename, pids=pids, header=header) :: CovariateMatrix{Float64}
end

# subroutines to define AbstractArray
Base.size(x::CovariateMatrix)         = size(x.x)
Base.size(x::CovariateMatrix, d::Int) = size(x.x, d)

Base.length(x::CovariateMatrix) = prod(size(x.x))

Base.ndims(x::CovariateMatrix) = ndims(x.x)

Base.endof(x::CovariateMatrix) = length(x.x)

Base.eltype(x::CovariateMatrix) = eltype(x.x) 

Base.linearindexing(::Type{CovariateMatrix}) = Base.LinearFast()

# here we annotate return value with type of x
# this hack is **VERY** important for efficient indexing x from a BEDFile!
# experience is that compiler does not know return value of getindex(x::CovariateMatrix{T},...)!
# consequently, return value of getindex(::BEDFile,...) is **Any**
# annotating type here changes return value of getindex(::BEDFile,...) to T <: Float !!!
Base.getindex{T <: Float}(x::CovariateMatrix{T}, i::Int) = getindex(x.x, i)::T
Base.getindex{T <: Float}(x::CovariateMatrix{T}, i::Int, j::Int) = getindex(x.x, i, j)::T

# fix this to change index for both x, xt?
#function Base.setindex!(x::CovariateMatrix, i::Int)
#  setindex!(x.x, i), setindex!(x.xt, i)
#end

function Base.setindex!(x::CovariateMatrix, i::Int, j::Int)
    setindex!(x.x, i, j)
    setindex!(x.xt, j, i)
end

function Base.similar(x::CovariateMatrix, T::Real, dims::Dims)
    CovariateMatrix(zeros(T, dims), dims[2], zeros(T, reverse(dims)), ["V" * "$i" for i = 1:dims[2]])
end # function Base.similar

function Base.similar(x::CovariateMatrix)
    n,p = size(x)
    T   = eltype(x)
    CovariateMatrix(SharedArray(T, (n,p), pids=procs()), p, SharedArray(T, (p,n), pids=procs()), ["V" * "$i" for i = 1:p])
end # function Base.similar

display(x::CovariateMatrix) = display(x.x)
