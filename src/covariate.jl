# container for nongenetic covariates
immutable CovariateMatrix{T <: Float, V <: SharedMatrix} <: AbstractArray{T, 2}
    x  :: V 
    p  :: Int
    xt :: V 

    CovariateMatrix(x::SharedMatrix{T}, p, xt::SharedMatrix{T}) = new(x,p,xt)
end

function CovariateMatrix{T <: Float}(
    x  :: SharedMatrix{T},
    p  :: Int,
    xt :: SharedMatrix{T}
)
    CovariateMatrix{T, typeof(x)}(x, p, xt)
end

function CovariateMatrix(
    T        :: Type, 
    filename :: AbstractString;
    pids     :: DenseVector{Int} = procs(),
    header   :: Bool = false
)
    # first load data
    x_temp = readdlm(filename, T, header=header)

    # make SharedArray container for x_temp
    x = SharedArray(T, size(x_temp), init = S -> localindexes(S) = zero(T), pids=pids)
    copy!(x, x_temp)

    # do same for x'
    xt = SharedArray(T, reverse(size(x_temp)), init = S -> localindexes(S) = zero(T), pids=pids)
    transpose!(xt, x_temp)

    # save p for convenience
    p = size(x,2)

    return CovariateMatrix(x, p, xt)
end

# default to Float64 constructor
function CovariateMatrix(
    filename :: AbstractString;
    pids     :: DenseVector{Int} = procs(),
    header   :: Bool = false
)
    CovariateMatrix(Float64, filename, pids=pids, header=header)
end

# subroutines to define AbstractArray
Base.size(x::CovariateMatrix)         = size(x.x)
Base.size(x::CovariateMatrix, d::Int) = size(x.x, d)

Base.length(x::CovariateMatrix) = prod(size(x.x))

Base.ndims(x::CovariateMatrix) = ndims(x.x)

Base.endof(x::CovariateMatrix) = length(x.x)

Base.eltype(x::CovariateMatrix) = eltype(x.x) 

Base.linearindexing(::Type{CovariateMatrix}) = Base.LinearSlow()

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
    CovariateMatrix(zeros(T, dims), dims[2], zeros(T, reverse(dims)))
end # function Base.similar

function Base.similar(x::CovariateMatrix)
    n,p = size(x)
    T   = eltype(x)
    CovariateMatrix(SharedArray(T, (n,p), pids=procs()), p, SharedArray(T, (p,n), pids=procs()))
end # function Base.similar

display(x::CovariateMatrix) = display(x.x)
