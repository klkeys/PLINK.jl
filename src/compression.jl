function compress{T <: Float}(
    x :: DenseMatrix{T},
)
    # get size of matrix
    (n,p) = size(x)

    # get designated blocksizes
    blocksize = ((n-1) >> 2) + 1
    tblocksize = ((p-1) >> 2) + 1

    # output vector
    outlen = blocksize*p
    y = zeros(Int8, outlen+3)

    # first three bytes are fixed
    y[1] = MNUM1
    y[2] = MNUM2
    y[3] = ONE8

    # counters for bookkeeping
    ngeno = 0
    ybyte = 4   # start after magic numbers!

    # loop over all columns in x
    for j = 1:p

        # each column starts with fresh counter
        # ybyte is offset by first 3 bookkeeping bytes plus previous columns
        ngeno = 0
        yidx = 4 + (j-1)*blocksize
        ybyte = ZERO8

        # loop over cases
        for i = 1:n
            ngeno += 1
#            println("ngeno = $ngeno, ybyte = $ybyte, i = $i, j = $j, x[i,j] = $(x[i,j])")
            if T == Float64
#                y[yidx] = y[yidx] << 2 + bin64[x[i,j]]
#                ybyte = ybyte << 2 + bin64[x[i,j]]
                ybyte = bin64[x[i,j]] << 2*(ngeno-1) | ybyte
#                println("ybyte = $ybyte")
            else
#                y[yidx] = y[yidx] << 2 + bin32[x[i,j]]
                ybyte = bin32[x[i,j]] << 2*(ngeno-1) | ybyte
            end

            # reset whenver we reach 4 genotypes
            # OR when we reach end of a column
            if ngeno >= 4 || i == n
                y[yidx] = ybyte
                yidx += 1
                ngeno = 0
                ybyte = ZERO8
            end
        end
    end
    return y
end
