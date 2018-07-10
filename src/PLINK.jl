module PLINK

using StatsFuns
### idea from Julia Hoffimann Mendes to conditionally load OpenCL module
# if no OpenCL library is available, then issue a warning
# set "cl" variable to Void,
# then conditionally load GPU code based on value of "cl"
try
    using OpenCL
catch e
    warn("PLINK.jl cannot find an OpenCL library and will not load GPU functions correctly.")
    global cl = nothing
end
using DataFrames

import Base.size
import Base.==
import Base.isequal
import Base.mean
import Base.copy
import Base.getindex
import Base.length
import Base.ndims
import Base.display
import Base.convert
import Base.A_mul_B!
import Base.At_mul_B!
import Base.setindex!
#import Base.*

export BEDFile
export decompress_genotypes!, decompress_genotypes
export A_mul_B!, A_mul_B
export At_mul_B!, At_mul_B
#export *
export sumsq!, sumsq
export mean!, prec!
export maf
export getindex
export compress
export read_plink_data
export prednames

# constants used for decompression purposes
const ZERO8  = convert(Int8,0)
const ONE8   = convert(Int8,1)
const TWO8   = convert(Int8,2)
const THREE8 = convert(Int8,3)

# PLINK magic numbers
const MNUM1  = convert(Int8,108)
const MNUM2  = convert(Int8,27)

# old PLINK magic numbers
#const MNUM1  = convert(Int8, -17)
#const MNUM2  = convert(Int8,-85)

# typealias floating point operations
const Float = Union{Float32,Float64}
"""
This lookup table encodes the following PLINK format for genotypes:

- 00 is homozygous for allele 1
- 01 is heterozygous
- 10 is missing
- 11 is homozygous for allele 2

We will represent missing (NA) with NaN.
Further note that the bytes are read from right to left.
That is, if we label each of the 8 position as A to H, we would label backwards:

    01101100
    HGFEDCBA

and so the first four genotypes are read as follows:

    01101100
    HGFEDCBA

          AB   00  -> homozygote (first)        -> 0
        CD     11  -> other homozygote (second) -> 2
      EF       01  -> heterozygote (third)      -> 1
    GH         10  -> missing genotype (fourth) -> NaN

Finally, when we reach the end of a SNP (or if in individual-mode, the end of an individual),
then we skip to the start of a new byte (i.e. skip any remaining bits in that byte).
For a precise desceiption of PLINK BED files, see the file type reference in the [PLINK documentation](http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml).

The implementation here uses bitshifting and a bit threshold against `Int8` value 3 to interpret the compressed data.
The bitshifting trick works left-to-right, in contrast to the PLINK convention of reading right-to-left.
Thus, the map in PLINK.jl is slightly different:

- 00 -> 0 -> 0.0
- 11 -> 3 -> 2.0
- **10 -> 2 -> 1.0**
- **01 -> 1 -> NaN**
"""
const genofloat = [0.0, NaN, 1.0, 2.0]
const genoint   = [0 -99 1 2]

# dictionaries used in compression to convert floating point numbers to Int8 numbers
const bin32  = Dict{Float32, Int8}(0.0f0 => ZERO8, NaN32 => ONE8, 1.0f0 => TWO8, 2.0f0 => THREE8)
const bin64  = Dict{Float64, Int8}(0.0 => ZERO8, NaN => ONE8, 1.0 => TWO8, 2.0 => THREE8)

# preloaded GPU kernels
# these are merely long strongs that contain the code in the file
const gpucode64 = readstring(open(Pkg.dir() * "/PLINK/src/kernels/iht_kernels64.cl"))
const gpucode32 = readstring(open(Pkg.dir() * "/PLINK/src/kernels/iht_kernels32.cl"))

include("covariate.jl")
include("data.jl")
include("genomatrix.jl")
include("bedfile.jl")
include("compression.jl")
include("decompression.jl")
if cl != nothing
    include("gpu.jl")
end
include("linalg.jl")

end # end module PLINK
