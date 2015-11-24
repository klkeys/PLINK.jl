# GET THE VALUE OF A GENOTYPE IN A COMPRESSED MATRIX
# argument X is almost vacuous because it ensures no conflict with current Array implementations
# it becomes useful for accessing nongenetic covariates
function getindex(
    X         :: BEDFile,
    x         :: DenseVector{Int8},
    row       :: Int,
    col       :: Int,
    blocksize :: Int;
    interpret :: Bool = true,
    float32   :: Bool = false
)
    if col <= X.p
        genotype_block = x[(col-1)*blocksize + ((row - 1) >>> 2) + 1]
        k = 2*((row-1) & 3)
        genotype = (genotype_block >>> k) & THREE8
        interpret && float32 && return geno32[genotype + ONE8]
        interpret && return geno64[genotype + ONE8]
        return genotype
    else
        return X.x2[row,(col-X.p)]
    end
end

# default for getindex with BEDFile, to enable array-like indexing
getindex(x::BEDFile, row::Int, col::Int) = getindex(x, x.x, row, col, x.blocksize, interpret=true, float32=false)

function getindex(x::BEDFile, rowidx::BitArray{1}, colidx::BitArray{1})

    yn = sum(rowidx)
    yp = sum(colidx)
    yblock  = ((yn-1) >>> 2) + 1
    ytblock = ((yp-1) >>> 2) + 1

    y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock)
    yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock)
    y2  = x.x2[rowidx,colidx]
    y2t = y2'
    p2  = size(y2,2)

    return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2')
end

function getindex(x::BEDFile, rowidx::UnitRange{Int64}, colidx::BitArray{1})

    yn = length(rowidx)
    yp = sum(colidx)
    yblock  = ((yn-1) >>> 2) + 1
    ytblock = ((yp-1) >>> 2) + 1

    y  = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock)
    yt = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock)
    y2 = x.x2[rowidx,colidx]
    p2 = size(y2,2)

    return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2)
end

function getindex(x::BEDFile, rowidx::BitArray{1}, colidx::UnitRange{Int64})

    yn = sum(rowidx)
    yp = length(colidx)
    yblock  = ((yn-1) >>> 2) + 1
    ytblock = ((yp-1) >>> 2) + 1

    y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock)
    yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock)
    y2  = x.x2[rowidx,colidx]
    p2  = size(y2,2)
    y2t = y2'

    return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2t)
end


function getindex(x::BEDFile, rowidx::UnitRange{Int64}, colidx::UnitRange{Int64})

    yn = length(rowidx)
    yp = length(colidx)
    yblock  = ((yn-1) >>> 2) + 1
    ytblock = ((yp-1) >>> 2) + 1

    y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock)
    yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock)
    y2  = x.x2[rowidx,colidx]
    p2  = size(y2,2)
    y2t = y2'

    return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2t)
end


"""
    decompress_genotypes!(y,x,snp,means,invstds)


This function decompresses into `y` a standardized column (SNP) of a PLINK BED file housed in `x`.
Missing genotypes are sent to zero.

Arguments:

- `y` is the matrix to fill with (standardized) dosages.
- `x` is the BEDfile object that contains the compressed `n` x `p` design matrix.
- `snp` is the current SNP (predictor) to extract.
- `means` is an array of column means for `x`.
- `invstds` is an array of column precisions for `x`.
"""
function decompress_genotypes!(
    y       :: DenseVector{Float64},
    x       :: BEDFile,
    snp     :: Int,
    means   :: DenseVector{Float64},
    invstds :: DenseVector{Float64}
)
    m = means[snp]
    d = invstds[snp]
    t = zero(Float64)
    if snp <= x.p
        @inbounds for case = 1:x.n
            t       = getindex(x,x.x,case,snp,x.blocksize)
            y[case] = ifelse(isnan(t), 0.0, (t - m)*d)
        end
    else
        @inbounds for case = 1:x.n
            y[case] = (x.x2[case,(snp-x.p)] - m) * d
        end
    end
    return nothing
end


function decompress_genotypes!(
    y       :: DenseVector{Float32},
    x       :: BEDFile,
    snp     :: Int,
    means   :: DenseVector{Float32},
    invstds :: DenseVector{Float32}
)
    m = means[snp]
    d = invstds[snp]
    t = zero(Float32)
    if snp <= x.p
        @inbounds for case = 1:x.n
            t       = getindex(x,x.x,case,snp,x.blocksize, float32=true)
            y[case] = ifelse(isnan(t), 0.0f0, (t - m)*d)
        end
    else
        @inbounds for case = 1:x.n
            y[case] = (x.x2[case,(snp-x.p)] - m) * d
        end
    end
    return nothing
end


"""
    decompress_genotypes(x, snp, means, invstds [, shared=true, pids=procs()])

This function decompresses from `x` a column of genotypes corresponding to a single SNP from a PLINK BED file.
It returns the standardized column of decompressed genotypes.

Arguments:

- `x` is the `BEDFile` object that contains the compressed `n` x `p` design matrix.
- `snp` is the SNP to decompress.
- `means` is a vector of the column means of `x`.
- `invstds` is a vector of the column precisions of `x`.

Optional Arguments:

- `shared` controls whether or not to return a `SharedArray`. Defaults to `true` (return a SharedArray), otherwise `decompress_genotypes` returns an `Array`.
- `pids` controls the processes to which the output `SharedArray` is distributed. Defaults to `procs()`. `pids` has no effect if `shared = false`.

Output:

- A vector of type `SharedArray` (or `Array` when `shared = false`) containing standardized allele dosages from column `snp` of `BEDFile` object `x`.
"""
function decompress_genotypes(
    x       :: BEDFile,
    snp     :: Int,
    means   :: DenseVector{Float64},
    invstds :: DenseVector{Float64};
    shared  :: Bool = true,
    pids    :: DenseVector{Int} = procs()
)
    y = ifelse(shared, SharedArray(Float64, x.n, init = S -> S[localindexes(S)] = zero(Float64), pids=pids), zeros(Float64,x.n))
    decompress_genotypes!(y,x,snp,means,invstds)
    return y
end


function decompress_genotypes(
    x       :: BEDFile,
    snp     :: Int,
    means   :: DenseVector{Float32},
    invstds :: DenseVector{Float32};
    shared  :: Bool = true,
    pids    :: DenseVector{Int} = procs()
)
    y = ifelse(shared, SharedArray(Float32, x.n, init = S -> S[localindexes(S)] = zero(Float32), pids=pids), zeros(Float32,x.n))
    decompress_genotypes!(y,x,snp,means,invstds)
    return y
end


"""
    decompress_genotypes!(Y, x, means, invstds [, standardize=true])


Decompress and standardize genotypes from a `BEDFile` object `x` into a floating point matrix `Y`.
Use this function judiciously, since the memory demands from decompressing large portions of `x` can grow quite large.

Arguments:

- `Y` is the matrix to fill with decompressed genotypes.
- `x` is the `BEDFile` object that contains the compressed `n` x `p` design matrix.
- `means` is a vector of columns means of `x`.
- `invstds` is a vector of column precisions of `x`.

Optional Arguments:

- `standardize` is a `Bool` to control standardization of allele dosages. Defaults to `true`.
"""
function decompress_genotypes!(
    Y           :: DenseMatrix{Float64},
    x           :: BEDFile;
    pids        :: DenseVector{Int}     = procs(),
    means       :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids),
    invstds     :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids),
    standardize :: Bool = true,
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn = size(x,1)
    xp = size(x,2)

    # ensure dimension compatibility
    n == xn || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
    p <= xp || throw(DimensionMismatch("Y has more columns than x"))

    ## TODO: parallelize this for SharedArrays
    @inbounds for j = 1:xp
        if standardize
            m = means[j]
            s = invstds[j]
        end
        @inbounds for i = 1:n
            Y[i,j] = getindex(x,x.x,i,j,x.blocksize,interpret=true,float32=false)
            if standardize
                Y[i,j] = (Y[i,j] - m) * s
            end
        end
    end

    return nothing
end


function decompress_genotypes!(
    Y           :: DenseMatrix{Float32},
    x           :: BEDFile;
    pids        :: DenseVector{Int}     = procs(),
    means       :: DenseVector{Float32} = mean(Float32,x, shared=true, pids=pids),
    invstds     :: DenseVector{Float32} = invstd(x,means, shared=true, pids=pids),
    standardize :: Bool = true
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn = size(x,1)
    xp = size(x,2)

    # ensure dimension compatibility
    n == xn || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
    p <= xp || throw(DimensionMismatch("Y has more columns than x"))

    ## TODO: parallelize this for SharedArrays
    @inbounds for j = 1:xp
        if standardize
            m = means[j]
            s = invstds[j]
        end
        @inbounds for i = 1:n
            Y[i,j] = getindex(x,x.x,i,j,x.blocksize,interpret=true,float32=true)
            if standardize
                Y[i,j] = (Y[i,j] - m) * s
            end
        end
    end

    return nothing
end

"""
    decompress_genotypes!(Y, x, indices, means, invstds)

When `Y` is a matrix and `indices` is a `BitArray` or `Int` array that indexes the columns of `x`, then `decompress_genotypes!()` decompresses only a subset of `x`.
"""
function decompress_genotypes!(
    Y       :: DenseMatrix{Float64},
    x       :: BEDFile,
    indices :: BitArray{1};
    pids    :: DenseVector{Int}     = procs(),
    means   :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids),
    invstds :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids)
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn = x.n
    xp = size(x,2)

    # ensure dimension compatibility
    n == xn            || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
    p <= xp            || throw(DimensionMismatch("Y has more columns than x"))
    sum(indices) <= xp || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

    # counter to ensure that we do not attempt to overfill Y
    current_col = 0

    quiet = true
    ## TODO: parallelize this for SharedArrays
    @inbounds for snp = 1:xp

        # use this column?
        if indices[snp]

            # add to counter
            current_col += 1
            quiet || println("filling current column $current_col with snp $snp")

            # extract column mean, inv std
            m = means[snp]
            d = invstds[snp]

            if snp <= x.p
                @inbounds for case = 1:n
                    t = getindex(x,x.x,case,snp,x.blocksize)
                    Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
                    quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                end
            else
                @inbounds for case = 1:n
                    Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
                    quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                end
            end

            # quit when Y is filled
            current_col == p && return nothing
        end
    end
    return nothing
end


function decompress_genotypes!(
    Y       :: DenseMatrix{Float32},
    x       :: BEDFile,
    indices :: BitArray{1};
    pids    :: DenseVector{Int}     = procs(),
    means   :: DenseVector{Float32} = mean(Float32,x, shared=true, pids=pids),
    invstds :: DenseVector{Float32} = invstd(x,means, shared=true, pids=pids)
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn = x.n
    xp = size(x,2)

    # ensure dimension compatibility
    n == xn            || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
    p <= xp            || throw(DimensionMismatch("Y has more columns than x"))
    sum(indices) <= xp || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

    # counter to ensure that we do not attempt to overfill Y
    current_col = 0

    quiet = true
    ## TODO: parallelize this for SharedArrays
    @inbounds for snp = 1:xp

        # use this column?
        if indices[snp]

            # add to counter
            current_col += 1
            quiet || println("filling current column $current_col with snp $snp")

            # extract column mean, inv std
            m = means[snp]
            d = invstds[snp]

            if snp <= x.p
                @inbounds for case = 1:n
                    t = getindex(x,x.x,case,snp,x.blocksize, float32=true)
                    Y[case,current_col] = ifelse(isnan(t), 0.0f0, (t - m)*d)
                    quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                end
            else
                @inbounds for case = 1:n
                    Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
                    quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                end
            end

            # quit when Y is filled
            current_col == p && return nothing
        end
    end
    return nothing
end


function decompress_genotypes!(
    Y       :: DenseMatrix{Float64},
    x       :: BEDFile,
    indices :: DenseVector{Int};
    pids    :: DenseVector{Int}     = procs(),
    means   :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids),
    invstds :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids)
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn = x.n
    xp = size(x,2)

    # ensure dimension compatibility
    n == xn          || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
    p <= xp          || throw(DimensionMismatch("Y has more columns than x"))
    length(indices) <= p || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

    # counter to ensure that we do not attempt to overfill Y
    current_col = 0

    quiet = true
    @inbounds for snp in indices

        # add to counter
        current_col += 1
        quiet || println("filling current column $current_col with snp $snp")

        # extract column mean, inv std
        m = means[snp]
        d = invstds[snp]
        if snp <= x.p
            @inbounds for case = 1:n
                t = getindex(x,x.x,case,snp,x.blocksize)
                Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
                quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
            end
        else
            @inbounds for case = 1:n
                Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
            end
        end

        # quit when Y is filled
        current_col == p && return nothing
    end

    return nothing
end


function decompress_genotypes!(
    Y       :: DenseMatrix{Float32},
    x       :: BEDFile,
    indices :: DenseVector{Int};
    pids    :: DenseVector{Int}     = procs(),
    means   :: DenseVector{Float32} = mean(Float32,x, shared=true, pids=pids),
    invstds :: DenseVector{Float32} = invstd(x,means, shared=true, pids=pids)
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn = x.n
    xp = size(x,2)

    # ensure dimension compatibility
    n == xn              || throw(DimensionMismatch("column of Y is not of same length as column of uncompressed x"))
    p <= xp              || throw(DimensionMismatch("Y has more columns than x"))
    length(indices) <= p || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

    # counter to ensure that we do not attempt to overfill Y
    current_col = 0

    quiet = true
    @inbounds for snp in indices

        # add to counter
        current_col += 1
        quiet || println("filling current column $current_col with snp $snp")

        # extract column mean, inv std
        m = means[snp]
        d = invstds[snp]
        if snp <= x.p
            @inbounds for case = 1:n
                t = getindex(x,x.x,case,snp,x.blocksize)
                Y[case,current_col] = ifelse(isnan(t), 0.0f0, (t - m)*d)
                quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
            end
        else
            @inbounds for case = 1:n
                Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
            end
        end

        # quit when Y is filled
        current_col == p && return nothing
    end

    return nothing
end


"""
    decompress_genotypes!(Y, x, indices, mask_n [,pids=procs(), means=mean(Float64,x,shared=true,pids=pids), invstds=(x,means,shared=true,pids=pids)])

If called with a vector `mask_n` of `0`s and `1`s, then `decompress_genotypes!()` will decompress the columns of `x` indexed by `indices` into `Y` while masking the rows indexed by `mask_n`.
"""
function decompress_genotypes!(
    Y       :: DenseMatrix{Float64},
    x       :: BEDFile,
    indices :: BitArray{1},
    mask_n  :: DenseVector{Int};
    pids    :: DenseVector{Int}     = procs(),
    means   :: DenseVector{Float64} = mean(Float64,x, shared=true, pids=pids),
    invstds :: DenseVector{Float64} = invstd(x,means, shared=true, pids=pids)
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn = x.n
    xp = size(x,2)

    # ensure dimension compatibility
    n <= xn             || throw(DimensionMismatch("column dimension of of Y exceeds column dimension of uncompressed x"))
    n == length(mask_n) || throw(DimensionMismatch("bitmask mask_n indexes different number of columns than column dimension of Y"))
    p <= xp             || throw(DimensionMismatch("Y has more columns than x"))
    sum(indices) <= xp  || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

    # counter to ensure that we do not attempt to overfill Y
    current_col = 0

    quiet = true
    @inbounds for snp = 1:xp

        # use this column?
        if indices[snp]

            # add to counter
            current_col += 1
            quiet || println("filling current column $current_col with snp $snp")

            # extract column mean, inv std
            m = means[snp]
            d = invstds[snp]

            if snp <= x.p
                @inbounds for case = 1:n
                    if mask_n[case] == 1
                        t = getindex(x,x.x,case,snp,x.blocksize)
                        Y[case,current_col] = ifelse(isnan(t), 0.0, (t - m)*d)
                        quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                    else
                        Y[case,current_col] = 0.0
                    end
                end
            else
                @inbounds for case = 1:n
                    if mask_n[case] == 1
                        Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
                        quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                    else
                        Y[case,current_col] = 0.0
                    end
                end
            end

            # quit when Y is filled
            current_col == p && return nothing
        end
    end
    return nothing
end


function decompress_genotypes!(
    Y       :: DenseMatrix{Float32},
    x       :: BEDFile,
    indices :: BitArray{1},
    mask_n  :: DenseVector{Int};
    pids    :: DenseVector{Int}     = procs(),
    means   :: DenseVector{Float32} = mean(Float32,x, shared=true, pids=pids),
    invstds :: DenseVector{Float32} = invstd(x,means, shared=true, pids=pids)
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn = x.n
    xp = size(x,2)

    # ensure dimension compatibility
    n <= xn             || throw(DimensionMismatch("column dimension of of Y exceeds column dimension of uncompressed x"))
    n == length(mask_n) || throw(DimensionMismatch("bitmask mask_n indexes different number of columns than column dimension of Y"))
    p <= xp             || throw(DimensionMismatch("Y has more columns than x"))
    sum(indices) <= xp  || throw(DimensionMismatch("Vector 'indices' indexes more columns than are available in Y"))

    # counter to ensure that we do not attempt to overfill Y
    current_col = 0

    quiet = true
    @inbounds for snp = 1:xp

        # use this column?
        if indices[snp]

            # add to counter
            current_col += 1
            quiet || println("filling current column $current_col with snp $snp")

            # extract column mean, inv std
            m = means[snp]
            d = invstds[snp]

            if snp <= x.p
                @inbounds for case = 1:n
                    if mask_n[case] == 1
                        t = getindex(x,x.x,case,snp,x.blocksize, float32=true)
                        Y[case,current_col] = ifelse(isnan(t), 0.0f0, (t - m)*d)
                        quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                    else
                        Y[case,current_col] = 0.0f0
                    end
                end
            else
                @inbounds for case = 1:n
                    if mask_n[case] == 1
                        Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
                        quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                    else
                        Y[case,current_col] = 0.0f0
                    end
                end
            end

            # quit when Y is filled
            current_col == p && return nothing
        end
    end
    return nothing
end
