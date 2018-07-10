function decomp_genocol!{T <: Float}(
    Y     :: DenseVecOrMat{T},
    x     :: BEDFile{T},
    col   :: Int,
    snp   :: Int,
    m     :: T,
    d     :: T;
    quiet :: Bool = true
)
    @inbounds for case = 1:x.geno.n
        t = x.geno[case,snp]
        Y[case,col] = t == ONE8 ? zero(T) : (int2geno(x,t) - m)*d
        quiet || println("Y[$case,$col] = ", Y[case,col])
    end
    return nothing
end

function decomp_genocol!{T <: Float}(
    Y      :: DenseVecOrMat{T},
    x      :: BEDFile{T},
    mask_n :: DenseVector{Int},
    col    :: Int,
    snp    :: Int,
    m      :: T,
    d      :: T;
    quiet  :: Bool = true
)
    @inbounds for case = 1:x.geno.n
        if mask_n[case] == 1
            t = x[case,snp]
            Y[case,col] = t == ONE8 ? zero(T) : (int2geno(x,t) - m)*d
            quiet || println("Y[$case,$col] = ", Y[case,col])
        else
            Y[case,col] = zero(T)
        end
    end
    return nothing
end

function decomp_covarcol!{T <: Float}(
    Y     :: DenseVecOrMat{T},
    x     :: BEDFile{T},
    col   :: Int,
    covar :: Int,
    m     :: T,
    d     :: T;
    quiet :: Bool = true
)
    @inbounds for case = 1:x.geno.n
        Y[case,col] = (x.covar.x[case,(covar-x.geno.p)] - m) * d
        quiet || println("Y[$case,$col] = ", Y[case,col])
    end
end

function decomp_covarcol!{T <: Float}(
    Y      :: DenseVecOrMat{T},
    x      :: BEDFile{T},
    mask_n :: DenseVector{Int},
    col    :: Int,
    covar  :: Int,
    m      :: T,
    d      :: T;
    quiet  :: Bool = true
)
    @inbounds for case = 1:x.geno.n
        if mask_n[case] == 1
            t = x[case,snp]
            Y[case,col] = (x.covar.x[case,(covar-x.geno.p)] - m) * d
            quiet || println("Y[$case,$col] = ", Y[case,col])
        else
            Y[case,col] = zero(T)
        end
    end
end

"""
    decompress_genotypes!(Y, x, [, standardize=true])

Can also be called with a matrix `Y`, in which case all genotypes are decompressed.
Use this function judiciously, since the memory demands from decompressing large portions of `x` can grow quite large.
Use optional argument `standardize` to control standardization of allele dosages.
"""
function decompress_genotypes!{T <: Float}(
    Y       :: DenseMatrix{T},
    x       :: BEDFile{T};
    pids    :: DenseVector{Int} = procs(),
    quiet   :: Bool             = true
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn,xp = size(x)

    # ensure dimension compatibility
    @assert n == xn "column of Y is not of same length as column of uncompressed x"
    @assert p <= xp "Y has more columns than x"

    ## TODO: parallelize this for SharedArrays
    @inbounds for j = 1:xp
        m = x.means[j]
        s = x.precs[j]
        if j <= x.geno.p
            decomp_genocol!(Y, x, j, j, m, s, quiet=quiet)
        else
            decomp_covarcol!(Y, x, j, j, m, s, quiet=quiet)
        end
    end
    return nothing
end



"""
    decompress_genotypes!(Y, x, idx, means, invstds)

When `Y` is a matrix and `idx` is a `BitArray` or `Int` array that indexes the columns of `x`, then `decompress_genotypes!()` decompresses only a subset of `x`.
"""
function decompress_genotypes!{T <: Float}(
    Y     :: DenseMatrix{T},
    x     :: BEDFile{T},
    idx   :: BitArray{1};
    pids  :: DenseVector{Int} = procs(),
    quiet :: Bool             = true
)

    # get dimensions of matrix to fill
    (n,p)  = size(Y)
    xn, xp = size(x)

    # ensure dimension compatibility
    @assert n == xn "column of Y is not of same length as column of uncompressed x"
    @assert p <= xp "Y has more columns than x"
    @assert sum(idx) <= xp "Vector 'idx' indexes more columns than are available in Y"

    # counter to ensure that we do not attempt to overfill Y
    col = 0

    ## TODO: parallelize this for SharedArrays
    @inbounds for snp = 1:xp

        # use this column?
        if idx[snp]

            # add to counter
            col += 1
            quiet || println("filling current column $current_col with snp $snp")

            # extract column mean, precision
            m = x.means[snp]
            s = x.precs[snp]

            if snp <= x.geno.p
                decomp_genocol!(Y, x, col, snp, m, s, quiet=quiet)
            else
                decomp_covarcol!(Y, x, col, snp, m, s, quiet=quiet)
            end

            # quit when Y is filled
            col == p && return nothing
        end
    end
    return nothing
end



function decompress_genotypes!{T <: Float}(
    Y     :: DenseMatrix{T},
    x     :: BEDFile{T},
    idx   :: DenseVector{Int};
    pids  :: DenseVector{Int} = procs(),
    quiet :: Bool             = true
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn,xp = size(x)

    # ensure dimension compatibility
    @assert n == xn "column of Y is not of same length as column of uncompressed x"
    @assert p <= xp "Y has more columns than x"
    @assert length(idx) <= p "Vector 'idx' indexes more columns than are available in Y"

    # counter to ensure that we do not attempt to overfill Y
    col = 0

    @inbounds for snp in idx

        # add to counter
        col += 1
        quiet || println("filling current column $current_col with snp $snp")

        # extract column mean, precision
        m = x.means[snp]
        s = x.precs[snp]
        if snp <= x.geno.p
            decomp_genocol!(Y, x, col, snp, m, s, quiet=quiet)
        else
            decomp_covarcol!(Y, x, col, snp, m, s, quiet=quiet)
        end

        # quit when Y is filled
        col == p && return nothing
    end

    return nothing
end


"""
    decompress_genotypes!(Y, x, idx, mask_n [,pids=procs(), quiet=true])

If called with a vector `mask_n` of `0`s and `1`s, then `decompress_genotypes!()` will decompress the columns of `x` indexed by `idx` into `Y` while masking the rows indexed by `mask_n`.
"""
function decompress_genotypes!{T <: Float}(
    Y      :: DenseMatrix{T},
    x      :: BEDFile{T},
    idx    :: BitArray{1},
    mask_n :: DenseVector{Int};
    pids   :: DenseVector{Int} = procs(),
    quiet  :: Bool             = true
)

    # get dimensions of matrix to fill
    (n,p) = size(Y)
    xn,xp = size(x,2)

    # ensure dimension compatibility
    @assert n <= xn "column dimension of of Y exceeds column dimension of uncompressed x"
    @assert n == length(mask_n) "bitmask mask_n indexes different number of columns than column dimension of Y"
    @assert p <= xp "Y has more columns than x"
    @assert sum(idx) <= xp "Vector 'idx' indexes more columns than are available in Y"

    # counter to ensure that we do not attempt to overfill Y
    col = 0

    @inbounds for snp = 1:xp

        # use this column?
        if idx[snp]

            # add to counter
            col += 1
            quiet || println("filling current column $col with snp $snp")

            # extract column mean, precision
            m = x.means[snp]
            s = x.precs[snp]
            if snp <= x.geno.p
                decomp_genocol!(Y, x, mask_n, col, snp, m, s, quiet=quiet)
            else
                decomp_covarcol!(Y, x, mask_n, col, snp, m, s, quiet=quiet)
            end

            # quit when Y is filled
            col == p && return nothing
        end
    end
    return nothing
end


"""
    decompress_genotypes!(y, x, col [, quiet=true])


This function decompresses into vector `y` a standardized column (SNP) of a PLINK BED file housed in `x`.
Missing genotypes are sent to zero.

Arguments:

- `y` is the vector to fill with (standardized) dosages.
- `x` is the BEDfile object that contains the compressed `n` x `p` design matrix.
- `col` is the current predictor to extract.

Optional Arguments:

- `quiet` is a `Bool` to activate output. Defaults to `true` (no output). Used mostly for debugging.
"""
function decompress_genotypes!{T <: Float}(
    y     :: DenseVector{T},
    x     :: BEDFile{T},
    col   :: Int;
    quiet :: Bool = true
)
    m = x.means[col]
    s = x.precs[col]
    if col <= x.geno.p
        decomp_genocol!(y, x, 1, col, m, s, quiet=quiet)
    else
        decomp_covarcol!(y, x, 1, col, m, s, quiet=quiet)
    end
    return nothing
end
