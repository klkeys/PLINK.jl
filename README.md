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

    immutable BEDFile{T <: Float, V <: SharedMatrix}
        geno  :: GenoMatrix
        covar :: CovariateMatrix{T, V}
        means :: SharedVector{T}
        precs :: SharedVector{T}
    end

At its core, a `BEDFile` is a fancy container object.
The `GenoMatrix` type actually houses the compressed genotype data:

    immutable GenoMatrix
        x          :: SharedVector{Int8}
        xt         :: SharedVector{Int8}
        n          :: Int
        p          :: Int
        blocksize  :: Int
        tblocksize :: Int
    end

The field `x` houses the `.bed` file, while `xt` contains the *transposed* `.bed` file.
Including transposed `.bed` files in a `BEDFile` object facilitates certain linear algebra operations.
For example, the matrix-vector operation `x * y` entails computing dot products along the rows of `x`.
Maintaining a transposed `.bed` file in memory yields faster operations than repeated decompressions of the rows of `x`.

The `CovariateMatrix` parametrix type does the same, but it houses nongenetic covariates in floating point form:

    immutable CovariateMatrix{T <: Float, V <: SharedMatrix} <: AbstractArray{T, 2}
        x  :: V  
        p  :: Int
        xt :: V  
    end

The fields `x`, `p`, and `xt` only matter if the analysis includes nongenetic covariates.
One example is regression analysis; users should put the grand mean (vector of ones) in a `CovariateMatrix`. 

`BEDFile` objects make heavy use of Julia `SharedArray`s to distribute computations across multiple CPU cores.
A wise practice is to maintain a common vector of process ids (e.g. `pids = procs()`).
Several functions in PLINK.jl can read `pids` as an optional argument.

There are many `BEDFile` constructors. Most users will use either of

    x = BEDFile("PATH_TO_BED.bed", "PATH_TO_TBED.bed")
    x = BEDFile("PATH_TO_BED.bed", "PATH_TO_TBED.bed", "PATH_TO_COVARIATES.txt")
 
depending on whether or not covariates are included.

## Standardizing

To facilitate linear algebra and regression analysis, PLINK.jl defaults to using standardized copies of `x`.
Since a standardized `x` cannot be stored in PLINK format, PLINK.jl offers facilities to calculates means and precisions (inverse standard deviations).
These quantities are stored in the fields `means` and `precs`:

    mean!(x) # calculate means and store in x.means
    prec!(x) # calculate precisions and store in x.precs

These calculations are not necessarily fast, so it is recommended that users precompute and cache the column means and precisions of a genotype matrix. 
If the means and precisions are stored to file, then a `BEDFile` with means and precisions can be constructed directly with

   x = BEDFile("PATH_TO_BED.bed", "PATH_TO_TBED.bed", "PATH_TO_COVARIATES.txt", "PATH_TO_MEANS.bin", "PATH_TO_PRECS.bin") 

Observe that the means and precisions must be stored in _binary_ format.
This constraint arises from the nature of the `SharedArray` constructor,
which uses a memory map.
If means and precisions are stored in a delimited file, then the user can load them and fill the corresponding fields manually:

    m = readdlm("PATH_TO_MEANS.txt")
    d = readdlm("PATH_TO_PRECS.txt")
    copy!(x.means, m)
    copy!(x.precs, d)

**IMPORTANT**: the onus is on the user to ensure that the means and precisions are reasonable.
For the regression analysis case in particular,
the user must set elements of `x.means` and `x.precs` to `0.0` and `1.0`, respectively.
PLINK.jl does *not* do this automatically! 

## Decompression

`BEDFile` objects have some matrix-like behavior and admit functions such as `size`, `length`, and `getindex`.
The latter enables selection of one genotype at a time, such as `x[1,1]`. PLINK.jl also permits decompression of entire columns or sets of columns at a time.

    y = zeros(x.n)
    decompress_genotypes!(y, x, 1, pids=pids)
    Y = zeros(x.n,2)
    idx = collect(1:2)                          # index with Int vector
    idb = falses(size(x,2)); idb[1:2] = true    # can also index with BitArray
    decompress_genotypes!(Y, x, idx, pids=pids)
    decompress_genotypes!(Y, x, idb, pids=pids)

It is possible to recover the floating point representation of `x`:

    Y = zeros(x.n, size(x,2))
    decompress_genotypes(Y, x, pids=pids)

Unless it is absoluately necessary, or unless the data dimensions are not too large, use of `decompress_genotypes!` in this way is strongly discouraged since the memory demands can balloon quickly.

## Linear algebra

In addition to the `mean!` and `prec!` functions decribed previously, PLINK.jl currently implements the following linear algebra functions:

* `sumsq` for squared L2 norms of the columns
* `At_mul_B!` and `At_mul_B` for `x' * y`
* `A_mul_B!` and `A_mul_B` for `x * b` 

`At_mul_B!` contains a parallel execution kernels modeled on the `advection_shared!` example from the [Julia parallel computing documentation](http://docs.julialang.org/en/latest/manual/parallel-computing/#id2).
Currently `A_mul_B!` is not parallelized because it is optimized for sparse vector multiplicands; consequently, the parallel overhead can often be greater than the actual amount of computation.
This feature could could change in the future.

## GPU acceleration

For `.bed` files with many genetic predictors, `At_mul_B!` becomes a serious computational bottleneck.
PLINK.jl uses [OpenCL wrappers](https://github.com/JuliaGPU/OpenCL.jl) to port `At_mul_B!` to a GPU.
The parallelization scheme uses two GPU kernels.
The first distributes a chunk of `x' * y` to a device workgroup.
Each thread in the workgroup decompresses, standardizes, and computes one component of `x' * y`.
The intermediate results are stored in a device buffer.
The second kernel then reduces along the buffer and returns the vector `x' * y` to the host. 
GPU use is an advanced topic, and `At_mul_B!` requires many additional arguments to work correctly on the GPU.
Interested users should be comfortable reading OpenCL and `At_mul_B!` source code.
