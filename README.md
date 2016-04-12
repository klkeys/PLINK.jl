# PLINK.jl

A module to handle [PLINK binary genotype files](http://pngu.mgh.harvard.edu/~purcell/plink/data.shtml#bed) in Julia.
This package furnishes decompression and compression routines for PLINK `.bed` files,
The compressed genotype matrix `x` is stored as a string of `Int8` components, where each 8-bit integer stores up to four genotypes. 
PLINK.jl also provides linear algebra routines that decompress `x` on the fly, including both `x * y` and `x' * y`. 

Future development will target the [PLINK 2 format](https://www.cog-genomics.org/plink2/input#bed).

## Download

At the Julia prompt, type

    Pkg.clone("https://github.com/klkeys/PLINK.jl")

## Basic use

The core of PLINK.jl is the `BEDFile` object:

    x   :: SharedVector{Int8}   # the BED file
    xt  :: SharedVector{Int8}   # the _transposed_ BED file
    n   :: Int                  # the number of cases (from FAM file)
    p   :: Int                  # the number of genotype predictors (from BIM file)
    blocksize  :: Int           # number of bytes required to store a column in X 
    tblocksize :: Int           # number of bytes required to store a row of X (col of X')
    x2  :: SharedArray          # array of nongenetic covariates
    p2  :: Int                  # number of nongenetic covariates
    x2t :: SharedArray          # transpose of nongenetic covariates

Including transposed `.bed` files in a `BEDFile` object facilitates certain linear algebra operations.
For example, the matrix-vector operation `x * y` entails computing dot products along the rows of `x`.
Maintaining a transposed `.bed` file in memory yields faster operations than repeated decompressions of the rows of `x`.
The fields `x2`, `p2`, and `x2t` usually only matter if the analysis includes nongenetic covariates,
though proper regression analysis should put the grand mean (vector of ones) in these slots.

`BEDFile` objects make heavy use of Julia `SharedArray`s to distribute computations across multiple CPU cores. A wise practice is to maintain a vector of process ids (e.g. `pids = procs()`). Several functions in PLINK.jl can read `pids` as an optional argument.

There are many `BEDFile` constructors. Most users will use either of

    x = BEDFile("PATH_TO_BED.bed", "PATH_TO_TBED.bed")
    x = BEDFile("PATH_TO_BED.bed", "PATH_TO_TBED.bed", "PATH_TO_COVARIATES.txt")
 
depending on whether or not covariates are included. Covariates can always be added afterwards:

    x  = BEDFile("PATH_TO_BED.bed", "PATH_TO_TBED.bed")
    x2 = SharedArray(Float64, (x.n,), init = S -> S[localindexes(S)] = 1.0)
    addx2!(x, x2)

## Standardizing

PLINK.jl coerces the user to use standardized copies of `x`. Since a standardized `x` cannot be stored in PLINK format, PLINK.jl calculates means and inverse standard deviations (precisions):

    m = mean(x)
    d = invstd(x, m)

Both `m` and `d` are optional arguments for the linear algebra routines,
but their omission will result in calculation of means and precisions on-the-fly.
These calculations are *not* fast, so it is recommended that users precompute and cache means and precisions. 

## Decompression

`BEDFile` objects have some matrix-like behavior and admit functions such as `size`, `length`, and `getindex`. The latter enables selection of one genotype at a time, such as `x[1,1]`. PLINK.jl also permits decompression of entire columns or sets of columns at a time.

    y = zeros(x.n)
    decompress_genotypes!(y, x, 1, pids=pids, means=m, invstds=d) 
    Y = zeros(x.n,2)
    idx = collect(1:2)                          # index with Int vector
    idb = falses(size(x,2)); idb[1:2] = true    # can also index with BitArray
    decompress_genotypes!(Y, x, idx, pids=pids, means=m, invstds=d)
    decompress_genotypes!(Y, x, idb, pids=pids, means=m, invstds=d)

It is possible to recover the floating point representation of `x`:

    Y = zeros(x.n, size(x,2))
    decompress_genotypes(Y, x, pids=pids, means=m, invstds=d [, standardize=true])

Use of `decompress_genotypes!` in this way is strongly discouraged since the memory demands can quickly breach computer memory limits. 


## Linear algebra

In addition to the `mean` and `invstd` functions decribed previously, 
PLINK.jl currently implements the following linear algebra functions:

    * `sumsq` for squared L2 norms of the columns
    * `xty!` for `x' * y`
    * `xb!` for `x * b` 

Both `xty!` and `xb!` contain parallel execution kernels modeled on `pmap`; see the [parallel computing documentation](http://docs.julialang.org/en/latest/manual/parallel-computing/#scheduling) for more details.

## GPU acceleration

For `.bed` files with many genetic predictors, `xty!` becomes a serious computational bottleneck.
PLINK.jl uses [OpenCL wrappers](https://github.com/JuliaGPU/OpenCL.jl) to port `xty!` to a GPU.
The parallelization scheme uses two GPU kernels.
The first distributes a chunk of `x' * y` to a device workgroup.
Each thread in the workgroup decompresses, standardizes, and computes one component of `x' * y`.
The intermediate results are stored in a device buffer.
The second kernel then reduces along the buffer and returns the vector `x' * y` to the host. 
GPU use is an advanced topic, and `xty!` requires many additional arguments to work correctly on the GPU.
Interested users should be comfortable reading OpenCL and `xty!` source code.
