"""
    getindex(X::BEDFile, x, x2, row, col)

The standard array access, coded for `BEDFile`s. Argument `X` is *almost* vacuous.
It primarily ensures no conflict with current Array implementations of `getindex`.
However, it becomes useful for accessing nongenetic covariates and checking the blocksize.
The default is to access the column-major PLINK file; for row-major access, use `X.xt` in argument `x`.
"""
function getindex{T <: Float}(
    X         :: BEDFile,
    x         :: DenseVector{Int8},
    x2        :: DenseVecOrMat{T},
    row       :: Int,
    col       :: Int,
)
    col > X.p && return X.x2[row,(col-X.p)]
    genotype_block = x[(col-1)*X.blocksize + ((row - 1) >>> 2) + 1]
    k = 2*((row-1) & 3)
    genotype = (genotype_block >>> k) & THREE8
    return int2geno[T][genotype + ONE8]
end

getindex(x::BEDFile, row::Int, col::Int) = getindex(x, x.x, x.x2, row, col)

### old code, never used?
#function getindex(x::BEDFile, rowidx::BitArray{1}, colidx::BitArray{1})
#
#    yn = sum(rowidx)
#    yp = sum(colidx)
#    yblock  = ((yn-1) >>> 2) + 1
#    ytblock = ((yp-1) >>> 2) + 1
#
#    y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock)
#    yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock)
#    y2  = x.x2[rowidx,colidx]
#    y2t = y2'
#    p2  = size(y2,2)
#
#    return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2')
#end
#
#function getindex(x::BEDFile, rowidx::UnitRange{Int64}, colidx::BitArray{1})
#
#    yn = length(rowidx)
#    yp = sum(colidx)
#    yblock  = ((yn-1) >>> 2) + 1
#    ytblock = ((yp-1) >>> 2) + 1
#
#    y  = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock)
#    yt = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock)
#    y2 = x.x2[rowidx,colidx]
#    p2 = size(y2,2)
#
#    return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2)
#end
#
#function getindex(x::BEDFile, rowidx::BitArray{1}, colidx::UnitRange{Int64})
#
#    yn = sum(rowidx)
#    yp = length(colidx)
#    yblock  = ((yn-1) >>> 2) + 1
#    ytblock = ((yp-1) >>> 2) + 1
#
#    y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock)
#    yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock)
#    y2  = x.x2[rowidx,colidx]
#    p2  = size(y2,2)
#    y2t = y2'
#
#    return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2t)
#end
#
#
#function getindex(x::BEDFile, rowidx::UnitRange{Int64}, colidx::UnitRange{Int64})
#
#    yn = length(rowidx)
#    yp = length(colidx)
#    yblock  = ((yn-1) >>> 2) + 1
#    ytblock = ((yp-1) >>> 2) + 1
#
#    y   = subset_genotype_matrix(x, x.x, rowidx, colidx, x.n, x.p, x.blocksize, yn=yn, yp=yp, yblock=yblock, ytblock=ytblock)
#    yt  = subset_genotype_matrix(x, x.xt, colidx, rowidx, x.p, x.n, x.tblocksize, yn=yp, yp=yn, yblock=ytblock, ytblock=yblock)
#    y2  = x.x2[rowidx,colidx]
#    p2  = size(y2,2)
#    y2t = y2'
#
#    return BEDFile(y,yt,yn,yp,yblock,ytblock,y2,p2,y2t)
#end


"""
    decompress_genotypes!(Y, x, means, invstds [, standardize=true])

Can also be called with a matrix `Y`, in which case all genotypes are decompressed.
Use this function judiciously, since the memory demands from decompressing large portions of `x` can grow quite large.
Use optional argument `standardize` to control standardization of allele dosages.
"""
function decompress_genotypes!{T <: Float}(
    Y           :: DenseMatrix{T},
    x           :: BEDFile;
    pids        :: DenseVector{Int} = procs(),
    means       :: DenseVector{T}   = mean(T,x, shared=true, pids=pids),
    invstds     :: DenseVector{T}   = invstd(x,means, shared=true, pids=pids),
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
            Y[i,j] = getindex(x, x.x, i, j)
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
function decompress_genotypes!{T <: Float}(
    Y       :: DenseMatrix{T},
    x       :: BEDFile,
    indices :: BitArray{1};
    pids    :: DenseVector{Int} = procs(),
    means   :: DenseVector{T}   = mean(T,x, shared=true, pids=pids),
    invstds :: DenseVector{T}   = invstd(x,means, shared=true, pids=pids)
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
                    t = getindex(x,x.x,case,snp)
                    Y[case,current_col] = ifelse(isnan(t), zero(T), (t - m)*d)
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



function decompress_genotypes!{T <: Float}(
    Y       :: DenseMatrix{T},
    x       :: BEDFile,
    indices :: DenseVector{Int};
    pids    :: DenseVector{Int}     = procs(),
    means   :: DenseVector{T} = mean(T,x, shared=true, pids=pids),
    invstds :: DenseVector{T} = invstd(x,means, shared=true, pids=pids)
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
                t = getindex(x,x.x,case,snp)
                Y[case,current_col] = ifelse(isnan(t), zero(T), (t - m)*d)
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
function decompress_genotypes!{T <: Float}(
    Y       :: DenseMatrix{T},
    x       :: BEDFile,
    indices :: BitArray{1},
    mask_n  :: DenseVector{Int};
    pids    :: DenseVector{Int} = procs(),
    means   :: DenseVector{T}   = mean(T,x, shared=true, pids=pids),
    invstds :: DenseVector{T}   = invstd(x,means, shared=true, pids=pids)
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
                        t = getindex(x,x.x,case,snp)
                        Y[case,current_col] = ifelse(isnan(t), zero(T), (t - m)*d)
                        quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                    else
                        Y[case,current_col] = zero(T)
                    end
                end
            else
                @inbounds for case = 1:n
                    if mask_n[case] == 1
                        Y[case,current_col] = (x.x2[case,(snp-x.p)] - m) * d
                        quiet || println("Y[$case,$current_col] = ", Y[case, current_col])
                    else
                        Y[case,current_col] = zero(T)
                    end
                end
            end

            # quit when Y is filled
            current_col == p && return nothing
        end
    end
    return nothing
end


"""
    decompress_genotypes!(y,x,snp,means,invstds)


This function decompresses into vector `y` a standardized column (SNP) of a PLINK BED file housed in `x`.
Missing genotypes are sent to zero.

Arguments:

- `y` is the matrix to fill with (standardized) dosages.
- `x` is the BEDfile object that contains the compressed `n` x `p` design matrix.
- `snp` is the current SNP (predictor) to extract.
- `means` is an array of column means for `x`.
- `invstds` is an array of column precisions for `x`.
"""
function decompress_genotypes!{T <: Float}(
    y       :: DenseVector{T},
    x       :: BEDFile,
    snp     :: Int,
    means   :: DenseVector{T},
    invstds :: DenseVector{T}
)
    m = means[snp]
    d = invstds[snp]
    t = zero(T)
    if snp <= x.p
        @inbounds for case = 1:x.n
            t = getindex(x,x.x,case,snp)
            y[case] = ifelse(isnan(t), zero(T), (t - m)*d)
        end
    else
        @inbounds for case = 1:x.n
            y[case] = (x.x2[case,(snp-x.p)] - m) * d
        end
    end
    return nothing
end
