"""
    mac(x::BEDFile) -> Vector{Int}

This function calculates the *m*inor *a*llele *c*ounts for each SNP of a `BEDFile` object `x`.
`mac` is similar to `maf` since it counts minor alleles, but it does not return the ratios of observed alleles. 
"""
function mac{T <: Float}(
    x :: BEDFile{T}
)
    z = zeros(Int, x.geno.p)
    @inbounds for i = 1:x.geno.p
        z[i] = mac_col(x, i)
    end
    return z
end

function mac_col{T <: Float}(
    x   :: BEDFile{T},
    col :: Int
)
    alleles = 0
    #obs     = 0
    for i = 1:x.geno.n
        dosage = x.geno[i,col]
        if dosage != 1 
            alleles += genoint[dosage + ONE8] 
            #obs += 1
        end
    end
    return alleles
end


"""
    maf(x::BEDFile) -> Vector{Float}

This function calculates the *m*inor *a*llele *f*requency for each SNP of a `BEDFile` object `x`.
"""
function maf{T <: Float}(
    x :: BEDFile{T}
)
    z = zeros(T, x.geno.p)
    @inbounds for i = 1:x.geno.p
        z[i] = maf_col(x, i)
    end
    return z
end

function maf_col{T <: Float}(
    x   :: BEDFile{T},
    col :: Int
)
    alleles = 0
    obs     = 0
    for i = 1:x.geno.n
        dosage = x.geno[i,col]
        if dosage != 1 
            alleles += genofloat[dosage + ONE8]
            obs += 1
        end
    end
    return alleles / (2*obs) 
end


"""
    sumsq_snp(x, snp)

This function efficiently computes the squared L2 (Euclidean) norm of column `snp` of a `BEDFile` object `x`, i.e. sumabs2(x[:,snp]).

Arguments:

- `x` is the `BEDFile` object containing the compressed `n` x `p` design matrix.
- `snp` is the current SNP (column) to use in calculations.
"""
function sumsq_snp{T <: Float}(x::BEDFile{T}, snp::Int)
    s = zero(T)       # accumulation variable, will eventually equal dot(y,z)
    t = zero(T)       # temp variable, output of interpret_genotype
    m = x.means[snp]
    d = x.precs[snp]

    # loop over all n individuals
    @inbounds for case = 1:x.geno.n
        t = x[case,snp]
        t = isnan(t) ? zero(T) : (t - m)*d
        s += t*t
    end
    return s
end



"""
    sumsq_covariate(x, covariate) 

This function efficiently computes the squared L2 (Euclidean) norm of a covariate of a `BEDFile` object `x`, i.e. sumabs2(x[:,covariate]).

Arguments:

- `x` is the `BEDFile` object containing the compressed `n` x `p` design matrix..
- `covariate` is the current nongenetic covariate (column) to use in calculations.
"""
function sumsq_covariate{T <: Float}(x::BEDFile{T}, covariate::Int)
    t = zero(T)
    s = zero(T)
    m = x.means[x.geno.p + covariate]
    d = x.precs[x.geno.p + covariate]

    # loop over all n individuals
    @inbounds for case = 1:x.geno.n
        t = (x.covar.x[case,covariate] - m) * d
        s += t*t
    end
    return s
end


"""
    sumsq(y, x)

Compute the squared L2 norm of each column of a compressed matrix `x` and save it to a vector `y`.

Arguments:

- `y` is the vector to fill with the squared norms.
- `x` is the `BEDFile` object that contains the compressed `n` x `p` design matrix from which to draw the columns.
"""
function sumsq!{T <: Float}(y::DenseVector{T}, x::BEDFile{T})
    p = size(x,2)
    p == length(y) || throw(DimensionMismatch("y has $(length(y)) rows, x has $p columns"))
    @inbounds for snp = 1:x.geno.p
        y[snp] = sumsq_snp(x,snp,x.means,x.precs) 
    end
    @inbounds for covariate = 1:x.covar.p
        y[x.geno.p + covariate] = sumsq_covariate(x,covariate,x.means,x.precs)
    end
    return nothing
end


"""
    sumsq(x::BEDFile [, shared=true, pids=procs()]) 

Compute the squared L2 norm of each column of a compressed matrix `x`.

Arguments:

- `x` is the `BEDFile` object that contains the compressed `n` x `p` design matrix from which to draw the columns.

Optional Arguments:
- `shared` is a `Bool` indicating whether or not to output a `SharedArray`. Defaults to `true`.
- `pids` is a vector of process IDs to which the output `SharedArray` will be distributed. Defaults to `procs()`. Has no effect when `shared = false`.
"""
function sumsq{T <: Float}(
    x       :: BEDFile{T};
    shared  :: Bool = true,
    pids    :: Vector{Int} = procs(),
)
    p = size(x,2)
    y = ifelse(shared, SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids), zeros(T, p))
    sumsq!(y,x)
    return y
end




### functions for column means

"""
    mean_chunk(x::BEDFile, irange)

A parallel execution kernel for calculating a subset of the column means of `x`.
"""
function mean_chunk!{T <: Float}(x::BEDFile{T}, irange::UnitRange{Int})
    for i in irange
        x.means[i] = mean_col(x, i)
    end
    return nothing
end

"""
    mean_chunk!(x::BEDFile)

A convenience wrapper for `mean_chunk!(q, x, irange)` that automatically chooses local indexes for `irange`.
"""
mean_chunk!{T <: Float}(x::BEDFile{T}) = mean_chunk!(x, localindexes(x.means))


"""
    mean_col(x::BEDFile, snp::Int)
    
Compute the mean of one `snp` column of a `BEDFile` object `x`. 
Note that unlike the normal Julia `mean` function, `mean_col` ignores `NaN`s.
"""
function mean_col{T <: Float}(x::BEDFile{T}, snp::Int)
    s = zero(T) # accumulation variable, will eventually equal mean(x,col) for current col
    t = zero(T) # temp variable, output of interpret_genotype
    u = zero(T) # count the number of people

    # loop over all n individuals
    @inbounds for case = 1:x.geno.n
        t = x[case,snp]

        # ensure that we do not count NaNs
        if isfinite(t)
            s += t
            u += one(T)
        end
    end

    # now divide s by u to get column mean and return
    return s /= u
end

"""
    mean!(x::BEDFile)

Compute the arithmetic means of the columns of a `BEDFile` object `x`.
Note that this function will ignore `NaN`s, unlike the normal Julia function `Base.mean`.

Arguments:

- `x` is the BEDFile object to use for computing column means.
"""
function Base.mean!{T <: Float}(x::BEDFile{T})
    # taken from advection example in Julia SharedArray documentation
    # this parallel execution structure distributes calculation of mean to all collaborating processors
    # each one computes an independent chunk of the mean vector based on localindexes()
    @sync begin
        for q in procs(x)
            @async remotecall_wait(mean_chunk!, q, x)
        end
    end
#    y[(x.geno.p+1):end] = vec(mean(x.covar.x,1))
    @inbounds for i = 1:x.covar.p
        x.means[x.geno.p + i] = mean(view(x.covar.x, :, i))
    end
    return nothing
end



### functions for precisions

"""
    prec_col(x::BEDFile, snp::Int)

Compute the precision of one `snp` column of a `BEDFile` object `x`.
"""
function prec_col{T <: Float}(x::BEDFile{T}, snp::Int)
    s = zero(T)      # accumulation variable, will eventually equal mean(x,col) for current col
    t = zero(T)      # temp variable, output of interpret_genotype
    u = zero(T)      # count the number of people
    m = x.means[snp] # column mean 

    # loop over all n individuals
    @inbounds for case = 1:x.geno.n
        t = x[case,snp]

        # ensure that we do not count NaNs
        if isfinite(t)
            s += (t - m)^2
            u += one(T)
        end
    end

    # now compute the std = sqrt(s / (u - 1)))
    s = s <= zero(T) ? zero(T) : sqrt((u - one(T)) / s)
    return s :: T
end

"""
    prec_chunk!(x::BEDFile, irange)

A parallel execution kernel for calculating a subset of column precisions of `x`. 
"""
function prec_chunk!{T <: Float}(x::BEDFile{T}, irange::UnitRange{Int})
    @inbounds for i in irange
        x.precs[i] = prec_col(x, i)
    end
    return nothing
end

"""
    prec_chunk!(x::BEDFile)

A convenience wrapper for `prec_chunk!(x, irange)` that automatically chooses local indexes for `irange`.
"""
prec_chunk!{T <: Float}(x::BEDFile{T}) = prec_chunk!(x, localindexes(x.precs))


"""
    prec!(x::BEDFile)

Compute the precision (inverse standard deviation) of the columns of a `BEDFile` object `x`.
The precisions are stored in `x.precs`.
Note that this function will ignore `NaN`s, unlike the normal Julia function `Base.std`.

Arguments:

- `x` is the `BEDFile` object to use for computing column standard deviations.
"""
function prec!{T <: Float}(x::BEDFile{T})
    # taken from advection example in Julia SharedArray documentation
    # this parallel execution structure distributes calculation of mean to all collaborating processors
    # each one computes an independent chunk of the mean vector based on localindexes()
    @sync begin
        for q in procs(x)
            @async remotecall_wait(prec_chunk!, q, x)
        end
    end
    @inbounds for i = 1:x.covar.p
        u = (one(T) / std(view(x.covar.x, :, i))) :: T
        x.precs[x.geno.p + i] = u 
    end
    return nothing
end




"""
    dot(x::BEDFile, y, snp)

This function computes the dot product of a column from the `BEDFile` object `x` against a vector `y`.

Arguments:

- `x` is the `BEDFile` object with the compressed `n` x `p` design matrix.
- `y` is the vector on which to perform the dot product.
- `snp` is the desired SNP (column) of the decompressed matrix to use for the dot product.

Optional Arguments:

- `sy = sum(y)`. `sy` is used for (efficient) standardization purposes. Typically this is computed once in the execution of `x' * y`.
"""
function Base.dot{T <: Float}(
    x       :: BEDFile{T},
    y       :: DenseVector{T},
    snp     :: Int,
    sy      :: T = sum(y)
)
    s = zero(T)        # accumulation variable, will eventually equal dot(y,z)
    m = x.means[snp]   # mean of SNP predictor
    d = x.precs[snp]   # 1/std of SNP predictor

    if snp <= x.geno.p

        # loop over all individuals
        @inbounds for case = 1:x.geno.n
            t = x.geno[case,snp]

            # handle exceptions on t
            u = t == ONE8 ? zero(T) : int2geno(x,t)

            # accumulate dot product
            s += y[case] * u
        end
    else
        @inbounds for case = 1:x.geno.n
            s += x.covar[case,snp-x.geno.p] * y[case]
        end
    end

    # return the (normalized) dot product
    return (s - m*sy)*d
end


"""
    dot(x,y,snp,mask_n [, sy=sum(y), sminus=(sum(y[mask_n .== 0])])

Can also be called with a bitmask vector `mask_n` containins `0`s and `1`s which removes masked rows of `x` and `y` from the dot product.
An additional keyword argument `sminus` encodes the number of `y`s equal to `0`.
"""
function Base.dot{T <: Float}(
    x       :: BEDFile{T},
    y       :: DenseVector{T},
    snp     :: Int,
    mask_n  :: DenseVector{Int},
    sy      :: T = sum(y),
    sminus  :: T = sum(y[mask_n .== 0]) # need this for standardization purposes
)
    s = zero(T)        # accumulation variable, will eventually equal dot(y,z)
    m = x.means[snp]   # mean of SNP predictor
    d = x.precs[snp]   # 1/std of SNP predictor

    if snp <= x.geno.p

        # loop over all individuals
        @inbounds for case = 1:x.geno.n

            # only accumulate if case is not masked
            if mask_n[case] == 1
                t = x.geno[case,snp]

                # handle exceptions on t
                u = t == ONE8 ? zero(T) : int2geno(x,t)

                # accumulate dot product
                s += y[case] * u 
            end
        end
    else
        @inbounds for case = 1:x.geno.n
            if mask_n[case] == 1
                s += x.covar.x[case,snp-x.geno.p] * y[case]
            end
        end
    end

    # return the (normalized) dot product
    return (s - (sy - sminus)*m)*d
end

function Base.dot{T <: Float}(
    x       :: BEDFile{T},
    y       :: DenseVector{T},
    snp     :: Int,
    mask_n  :: BitArray{1}, 
    sy      :: T = sum(y),
    sminus  :: T = sum(y[!mask_n]) # need this for standardization purposes
)
    s = zero(T)        # accumulation variable, will eventually equal dot(y,z)
    m = x.means[snp]   # mean of SNP predictor
    d = x.precs[snp]   # 1/std of SNP predictor

    if snp <= x.geno.p

        # loop over all individuals
        @inbounds for case = 1:x.geno.n

            # only accumulate if case is not masked
            if mask_n[case]
                t = x.geno[case,snp]

                # handle exceptions on t
                u = t == ONE8 ? zero(T) : int2geno(x,t)

                # accumulate dot product
                s += y[case] * u 
            end
        end
    else
        @inbounds for case = 1:x.geno.n
            if mask_n[case]
                s += x.covar.x[case,snp-x.geno.p] * y[case]
            end
        end
    end

    # return the (normalized) dot product
    return (s - (sy - sminus)*m)*d
end


"""
    dott(x::BEDFile, b, case, idx::BitArray{1})

This function computes the dot product of a row from the `BEDFile` object `x` against a vector `b`.
It computes `dot(x[case,:], b)` as `dot(x'[:,case], b)` to respect memory stride.

Arguments:

- `x` is the `BEDFile` object with the compressed `n` x `p` design matrix.
- `b` is the vector on which to perform the dot product.
- `case` is the desired case (row) of the decompressed matrix to use for the dot product.
- `idx` is a `BitArray` that indexes `b`.
"""
function dott{T <: Float}(
    x    :: BEDFile{T},
    b    :: DenseVector{T},
    case :: Int,
    idx  :: BitArray{1}
)
    s = zero(T)     # accumulation variable, will eventually equal dot(y,z)
    @inbounds for snp = 1:x.geno.p
        
        # if current index of b is FALSE, then skip it since it does not contribute to Xb
        if idx[snp]

            # decompress genotype, this time from transposed matrix
            t = getindex_t(x.geno,x.geno.xt,snp,case)

            # handle exceptions on t
            u = t == ONE8 ? zero(T) : (int2geno(x,t) - x.means[snp]) * x.precs[snp]
           
            # accumulate dot product
            s += b[snp] * u 
        end
    end
    @inbounds for covar = (x.geno.p+1):size(x,2)
        if idx[covar]
            s += b[covar] * (x.covar.xt[covar-x.geno.p,case] - x.means[covar]) * x.precs[covar]
        end
    end

    # return the dot product
    return s
end



"""
    A_mul_B!(xb, x, b, idx, k, mask_n [, pids=procs()])

Can also be called with a bitmask vector `mask_n` containins `0`s and `1`s which excludes or includes (respectively) elements of `x` and `b` from the dot product.
"""
function Base.A_mul_B!{T <: Float}(
    xb     :: DenseVector{T},
    x      :: BEDFile{T},
    b      :: DenseVector{T},
    idx    :: BitArray{1},
    k      :: Int,
    mask_n :: DenseVector{Int};
#    pids   :: Vector{Int} = procs(),
)
    # error checking
    0 <= k <= size(x,2) || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
    k >= sum(idx)       || throw(ArgumentError("Must have k >= sum(idx) or X*b will not compute correctly"))
    n = length(xb)
    n == x.geno.n       || throw(ArgumentError("xb has $n rows but x has $(x.geno.n) rows"))

    # loop over the desired number of predictors
    @inbounds for case = 1:n
        if mask_n[case] == 1
            xb[case] = dott(x, b, case, idx)
        end
    end

    return nothing
end

function Base.A_mul_B!{T <: Float}(
    xb     :: DenseVector{T},
    x      :: BEDFile{T},
    b      :: DenseVector{T},
    idx    :: BitArray{1},
    k      :: Int,
    mask_n :: BitArray{1}; 
#    pids   :: Vector{Int} = procs(x),
)
    # error checking
    0 <= k <= size(x,2) || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
    k >= sum(idx)       || throw(ArgumentError("Must have k >= sum(idx) or X*b will not compute correctly"))
    n = length(xb)
    n == x.geno.n       || throw(ArgumentError("xb has $n rows but x has $(x.geno.n) rows"))

    # loop over the desired number of predictors
    @inbounds for case = 1:n
        if mask_n[case]
            xb[case] = dott(x, b, case, idx)
        end
    end

    return nothing
end


"""
    A_mul_B!(xb, x, b, idx, k [, pids=procs()])

This function computes the operation `x*b` for the compressed `n` x `p` design matrix from a `BEDFile` object.
`A_mul_B!()` respects memory stride for column-major arrays by using a compressed transpose of genotypes and an uncompressed transpose of covariates.
It also assumes a sparse b, for which we have a `BitArray` index vector `idx` to select the nonzeroes.

Arguments:

- `xb` is the `n`-dimensional output vector.
- `x` is the `BEDFile` object for the compressed `n` x `p` design matrix.
- `b` is the `p`-dimensional vector against which we multiply `x`.
- `idx` is a `BitArray` that indexes the nonzeroes in `b`.
- `k` is the number of nonzeroes to use in computing `x*b`.

Optional Arguments:

- `pids` is a vector of process IDs over which to distribute the `SharedArray`s for `means` and `precs`, if not supplied. Defaults to `procs()`.
"""
function Base.A_mul_B!{T <: Float}(
    xb   :: DenseVector{T},
    x    :: BEDFile{T},
    b    :: DenseVector{T},
    idx  :: BitArray{1},
    k    :: Int;
#    pids :: Vector{Int} = procs(x),
)
    # error checking
    n = length(xb)
    n == x.geno.n       || throw(ArgumentError("xb has $n rows but x has $(x.geno.n)"))
    0 <= k <= size(x,2) || throw(ArgumentError("Number of active predictors must be nonnegative and less than p"))
    k >= sum(idx)       || throw(ArgumentError("Must have k <= sum(idx) or X*b will not compute correctly"))
#    pids == procs(xb) == procs(b) == procs(x.geno.xt) || throw(ArgumentError("SharedArray arguments to A_mul_B! must be seen by same processes"))

    # loop over the desired number of predictors
    @inbounds for case = 1:x.geno.n
        xb[case] = dott(x, b, case, idx)
    end

    return nothing
end



"""
    A_mul_B(x, b, idx, k, mask_n [, pids=procs()]) 

Can also be called with a bitmask vector `mask_n` containins `0`s and `1`s which excludes or includes (respectively) elements of `x` and `b` from the dot product.
"""
function A_mul_B{T <: Float}(
    x      :: BEDFile{T},
    b      :: SharedVector{T},
    idx    :: BitArray{1},
    k      :: Int,
    mask_n :: DenseVector{Int};
#    pids   :: Vector{Int} = procs(x),
)
    xb = SharedArray(T, x.geno.n, init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}
    A_mul_B!(xb, x, b, idx, k, mask_n, pids=pids)
    return xb
end

function A_mul_B{T <: Float}(
    x      :: BEDFile{T},
    b      :: Vector{T},
    idx    :: BitArray{1},
    k      :: Int,
    mask_n :: DenseVector{Int};
)
    xb = zeros(T, x.geno.n)
    A_mul_B!(xb, x, b, idx, k, mask_n) 
    return xb
end



"""
    A_mul_B(x::BEDFile, b, idx, k [, pids=procs()])

This function computes the operation `x*b` for the compressed `n` x `p` design matrix from a `BEDFile` object.
`A_mul_B()` respects memory stride for column-major arrays.
It also assumes a sparse `b`, for which we have a `BitArray` index vector `idx` to select the nonzeroes.
The output type matches the type of `b`.

Arguments:

- `x` is the `BEDFile` object for the compressed `n` x `p` design matrix.
- `b` is the `p`-dimensional vector against which we multiply `x`.
- `idx` is a `BitArray` that indexes the nonzeroes in `b`.
- `k` is the number of nonzeroes to use in computing `x*b`.

Optional Arguments:

- `pids` is a vector of process IDs over which to distribute the `SharedArray`s for `means` and `precs`, if not supplied,
   as well as the output vector. Defaults to `procs()`. Only available for `SharedArray` arguments to `b`.
"""
function A_mul_B{T <: Float}(
    x    :: BEDFile{T},
    b    :: SharedVector{T},
    idx  :: BitArray{1},
    k    :: Int;
#    pids :: Vector{Int} = procs(x),
)
    xb = SharedArray(T, x.geno.n, init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}
    A_mul_B!(xb, x, b, idx, k) 
    return xb
end

function A_mul_B{T <: Float}(
    x   :: BEDFile{T},
    b   :: Vector{T},
    idx :: BitArray{1},
    k   :: Int;
)
    xb = zeros(T, x.geno.n)
    A_mul_B!(xb, x, b, idx, k)
    return Xb
end

# overload matrix-vector multiplication x*b with a BEDFile
#Base.*{T <: Float}(x :: BEDFile{T}, b :: DenseVector{T}) = A_mul_B(x, b, b .!= zero(T), countnz(b), pids=procs(x))



"""
    At_mul_B_chunk!(xty, x::BEDFile, y, mask_n, irange)

A parallel execution kernel for calculating `x' * y` with a `BEDFile` object `x` that allows inclusion/exclusion of indices based on `mask_n`.
"""
function At_mul_B_chunk!{T <: Float}(
    xty    :: SharedVector{T},
    x      :: BEDFile{T},
    y      :: SharedVector{T},
    mask_n :: DenseVector{Int},
    irange :: UnitRange{Int},
    sy     :: T = sum(y),
    sminus :: T = begin z = mask_n .== 0; sum(y[z]) end
)
    @inbounds for i in irange
        xty[i] = dot(x, y, i, mask_n, sy, sminus) 
    end
    return nothing
end

function At_mul_B_chunk!{T <: Float}(
    xty    :: SharedVector{T},
    x      :: BEDFile{T},
    y      :: SharedVector{T},
    mask_n :: BitArray{1}, 
    irange :: UnitRange{Int},
    sy     :: T = sum(y),
    sminus :: T = sum(y[!mask_n])
)
    @inbounds for i in irange
        xty[i] = dot(x, y, i, mask_n, sy, sminus) 
    end
    return nothing
end

"""
    At_mul_B_chunk!(xty, x::BEDFile, y, mask_n)

A convenience wrapper for `At_mul_B_chunk!(xty, x, y, irange)` that automatically chooses local indexes for `irange` and excludes elements based on `mask_n`.
"""
function At_mul_B_chunk!{T <: Float}(
    xty    :: SharedVector{T},
    x      :: BEDFile{T},
    y      :: SharedVector{T},
    mask_n :: DenseVector{Int},
    sy     :: T = sum(y),
    sminus :: T = begin z = mask_n .== 0; sum(y[z]) end
)
    At_mul_B_chunk!(xty, x, y, mask_n, localindexes(xty), sy, sminus)
    return nothing
end

function At_mul_B_chunk!{T <: Float}(
    xty    :: SharedVector{T},
    x      :: BEDFile{T},
    y      :: SharedVector{T},
    mask_n :: BitArray{1}, 
    sy     :: T = sum(y),
    sminus :: T = sum(y[!mask_n])
)
    At_mul_B_chunk!(xty, x, y, mask_n, localindexes(xty), sy, sminus)
    return nothing
end


"""
    At_mul_B!(xty, x::BEDFile, y, mask_n, [, pids=procs()]) 

Can also be called with a bitmask vector `mask_n` containins `0`s and `1`s which excludes or includes elements of `x` and `y` from the dot product.
"""
function Base.At_mul_B!{T <: Float}(
    xty     :: SharedVector{T},
    x       :: BEDFile{T},
    y       :: SharedVector{T},
    mask_n  :: DenseVector{Int};
    pids    :: Vector{Int} = procs(),
    sy      :: T = sum(y),
    #sminus  :: T = sum(y[mask_n .== 0])
    sminus  :: T = begin z = mask_n .== 0; sum(y[z]) end
)
    # error checking
    p = size(x,2)
    p <= length(xty)      || throw(ArgumentError("Attempting to fill argument xty of length $(length(xty)) with $p elements!"))
    x.geno.n == length(y) || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.geno.n) of them!"))
    pids == procs(xty) == procs(y) == procs(x.geno.x) || throw(ArgumentError("SharedArray arguments to At_mul_B! must be seen by same processes"))

    # each processor will compute its own chunk of x'*y
    @sync begin
        for q in procs(xty)
#            @async remotecall_wait(At_mul_B_chunk!, q, xty, x, y, mask_n, sy, sminus) 
            @async remotecall_wait(At_mul_B_chunk!, q, xty, x, y, mask_n, sy, sminus)
        end
    end
    return nothing
end

function Base.At_mul_B!{T <: Float}(
    xty     :: SharedVector{T},
    x       :: BEDFile{T},
    y       :: SharedVector{T},
    mask_n  :: BitArray{1}; 
    pids    :: Vector{Int} = procs(x),
    sy      :: T = sum(y),
    #sminus  :: T = sum(y[mask_n .== 0])
    sminus  :: T = begin z = !mask_n; sum(y[z]) end
)
    # error checking
    p = size(x,2)
    p <= length(xty)      || throw(ArgumentError("Attempting to fill argument xty of length $(length(xty)) with $p elements!"))
    x.geno.n == length(y) || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.geno.n) of them!"))
    pids == procs(xty) == procs(y) == procs(x.geno.x) || throw(ArgumentError("SharedArray arguments to At_mul_B! must be seen by same processes"))

    # each processor will compute its own chunk of x'*y
    @sync begin
        for q in procs(xty)
#            @async remotecall_wait(At_mul_B_chunk!, q, xty, x, y, mask_n, sy, sminus) 
            @async remotecall_wait(At_mul_B_chunk!, q, xty, x, y, mask_n, sy, sminus)
        end
    end
    return nothing
end


function Base.At_mul_B!{T <: Float}(
    xty     :: Vector{T},
    x       :: BEDFile{T},
    y       :: Vector{T},
    mask_n  :: DenseVector{Int},
    sy      :: T = sum(y),
    #sminus  :: T = sum(y[mask_n .== 0])
    sminus  :: T = begin z = mask_n .== 0; sum(y[z]) end
)
    # error checking
    p = size(x,2)
    p <= length(xty)      || throw(ArgumentError("Attempting to fill argument xty of length $(length(xty)) with $p elements!"))
    x.geno.n == length(y) || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.geno.n) of them!"))

    # loop over the desired number of predictors
    @inbounds for snp = 1:p
        xty[snp] = dot(x, y, snp, mask_n, sy, sminus) 
    end
    return nothing
end

function Base.At_mul_B!{T <: Float}(
    xty     :: Vector{T},
    x       :: BEDFile{T},
    y       :: Vector{T},
    mask_n  :: BitArray{1}, 
    sy      :: T = sum(y),
    #sminus  :: T = sum(y[mask_n .== 0])
    sminus  :: T = begin z = !mask_n; sum(y[z]) end
)
    # error checking
    p = size(x,2)
    p <= length(xty)      || throw(ArgumentError("Attempting to fill argument xty of length $(length(xty)) with $p elements!"))
    x.geno.n == length(y) || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.geno.n) of them!"))

    # loop over the desired number of predictors
    @inbounds for snp = 1:p
        xty[snp] = dot(x, y, snp, mask_n, sy, sminus) 
    end
    return nothing
end


"""
    At_mul_B_chunk!(xty, x::BEDFile, y, irange)

A parallel execution kernel for calculating `x' * y` with a `BEDFile` object `x`.
"""
function At_mul_B_chunk!{T <: Float}(
    xty     :: SharedVector{T},
    x       :: BEDFile{T},
    y       :: SharedVector{T},
    irange  :: UnitRange{Int},
    sy      :: T = sum(y),
)
    @inbounds for i in irange
        xty[i] = dot(x, y, i, sy)
    end
    return nothing
end

"""
    At_mul_B_chunk!(xty, x::BEDFile, y)

A convenience wrapper for `At_mul_B_chunk!(xty, x, y, irange)` that automatically chooses local indexes.
"""
function At_mul_B_chunk!{T <: Float}(
    xty     :: SharedVector{T},
    x       :: BEDFile{T},
    y       :: SharedVector{T},
    sy      :: T = sum(y)
)
    At_mul_B_chunk!(xty, x, y, localindexes(xty), sy)
    return nothing
end

"""
    At_mul_B!(xty, x, y, [, pids=procs()]) 

This function computes the operation `x'*y` for the compressed `n` x `p` design matrix from a `BEDFile` object.
`At_mul_B!` enforces a uniform type (`SharedArray` v. `Array` and `Float64` v. `Float32`) across all arrays.
It also requires `SharedArray` arguments to be seen by the same processes, i.e. `procs(xty) == procs(y) == procs(x)`.

Arguments:

- 'xty' is the p-dimensional output vector.
- `x` is the `BEDFile` object for the compressed `n` x `p` design matrix.
- `y` is the `n`-dimensional vector against which we multiply `x`.

Optional Arguments:

- `pids` is a vector of process IDs over which to distribute the `SharedArray`s for `means` and `precs`, if not supplied,
   as well as the output vector. Defaults to `procs()`. Only available for `SharedArray` arguments to `xty` and `y`.
"""
function Base.At_mul_B!{T <: Float}(
    xty     :: SharedVector{T},
    x       :: BEDFile{T},
    y       :: SharedVector{T};
    pids    :: Vector{Int} = procs(),
    sy      :: T = sum(y)
)
    # error checking
    p = size(x,2)
    p <= length(xty) || throw(ArgumentError("Attempting to fill argument xty of length $(length(xty)) with $p elements!"))
    x.geno.n == length(y) || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.n) of them!"))
    pids == procs(xty) == procs(y) == procs(x) || throw(ArgumentError("SharedArray arguments to At_mul_B! must be seen by same processes"))

    # each processor will compute its own chunk of x'*y 
    @sync begin
        for q in procs(xty)
#            @async remotecall_wait(At_mul_B_chunk!, q, xty, x, y, sy)
            @async remotecall_wait(At_mul_B_chunk!, q, xty, x, y, sy)
        end
    end

    return nothing
end



function Base.At_mul_B!{T <: Float}(
    xty     :: Vector{T},
    x       :: BEDFile{T},
    y       :: Vector{T};
    sy      :: T   = sum(y),
)
    # error checking
    p = size(x,2)
    p <= length(xty) || throw(ArgumentError("Attempting to fill argument xty of length $(length(xty)) with $p elements!"))
    x.geno.n == length(y)   || throw(ArgumentError("Argument y has $(length(y)) elements but should have $(x.geno.n) of them!"))

    # loop over the desired number of predictors
    @inbounds for snp = 1:p
        xty[snp] = dot(x, y, snp, sy)
    end
    return nothing
end



"""
    At_mul_B(x, y, mask_n [, pids=procs()])

This function computes `x'*y` for the compressed `n` x `p` design matrix from a `BEDFile` object.
It uses a bitmask `mask_n` to exclude certain rows of `x` from the calculations.

Arguments:

- `x` is the `BEDFile` object for the compressed `n` x `p` design matrix.
- `y` is the `n`-dimensional vector against which we multiply `x`.
- `mask_n` is an `n`-vector of `0`s and `1`s that indicates which rows to exclude or include, respectively.

Optional Arguments:

- `pids` is a vector of process IDs over which to distribute the `SharedArray`s for `means` and `precs`, if not supplied,
   as well as the output vector. Defaults to `procs()`. Only available for `SharedArray` arguments to `b`.
"""
function Base.At_mul_B{T <: Float}(
    x       :: BEDFile{T},
    y       :: SharedVector{T},
    mask_n  :: DenseVector{Int};
    pids    :: Vector{Int} = procs(),
    sy      :: T = sum(y)
)
    p = size(x,2) 
    xty = SharedArray(T, p, init = S -> S[localindexes(S)] = zero(T), pids=pids) :: SharedVector{T}
    At_mul_B!(xty, x, y, mask_n, pids=pids, sy=sy)
    return xty
end


function Base.At_mul_B{T <: Float}(
    x       :: BEDFile{T},
    y       :: Vector{T},
    mask_n  :: DenseVector{Int}; 
    sy      :: T = sum(y),
    sminus  :: T = sum(y[mask_n .== 0])
)
    p = size(x,2) 
    xty = zeros(T, p)
    At_mul_B!(xty, x, y, mask_n, sy, sminus) 
    return xty
end
