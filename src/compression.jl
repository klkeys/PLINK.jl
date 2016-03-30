function compress{T <: Float}(
    x :: DenseMatrix{T}
)
    # get size of matrix
    (n,p) = size(x)

    # get designated blocksize
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
    ybyte = 4

    # loop over all columns in x
    for j = 1:p

        # each column starts with fresh counter
        # ybyte is offset by first 3 bookkeeping bytes plus previous columns
        ngeno = 0
        ybyte = 4 + (j-1)*blocksize

        # loop over cases
        for i = 1:n
            ngeno += 1
#            println("ngeno = $ngeno, ybyte = $ybyte, i = $i, j = $j")
            if T == Float64
                y[ybyte] = y[ybyte] << 2 + bin64[x[i,j]]
            else
                y[ybyte] = y[ybyte] << 2 + bin32[x[i,j]]
            end

            # reset whenver we reach 4 genotypes
            if ngeno >= 4
                ybyte += 1
                ngeno = 0
            end
        end
    end
    return y
end

#function compress(
#    x :: DenseMatrix{Integer}
#)
#    # get size of matrix
#    (n,p) = size(x)
#
#    # get designated blocksize
#    blocksize = ((n-1) >> 2) + 1
#    tblocksize = ((p-1) >> 2) + 1
#
#    # output vector
#    outlen = blocksize*p
#    y = zeros(Int8, outlen+3)
#
#    # first three bytes are fixed
#    y[1] = MNUM1
#    y[2] = MNUM2
#    y[3] = ONE8
#
#    # counters for bookkeeping
#    ngeno = 0
#    ybyte = 4
#
#    # loop over all columns in x
#    for j = 1:p
#
#        # each column starts with fresh counter
#        # ybyte is offset by first 3 bookkeeping bytes plus previous columns
#        ngeno = 0
#        ybyte = 4 + (j-1)*blocksize
#
#        # loop over cases
#        for i = 1:n
#            ngeno += 1
##            println("ngeno = $ngeno, ybyte = $ybyte, i = $i, j = $j")
#            y[ybyte] = y[ybyte] << 2 + binint[x[i,j]]
#
#            # reset whenver we reach 4 genotypes
#            if ngeno >= 4
#                ybyte += 1
#                ngeno = 0
#            end
#        end
#    end
#    return y
#end
