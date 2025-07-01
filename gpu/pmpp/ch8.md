stencil
- this is an iterative computation
- there are some dependencies
- also higher numerical accuracies
# background
- numerical evaluation and solving requires converting to discrete representation
- derivatives can be finite differences, so we use structured grids in finite-difference methods
- discrete rep: use interpolation (linear, splines) to derive approximate value of the function
- smaller spacing means higher accuracy, but more memory and compute
- also different levels of precision
- what is a stencil?
	- geometric pattern of weights applied at each point of a structured grid
	- pattern specifies how values of grid point of interest can be derived from values at neighboring points using numerical approximation routine
- number of grid points on each side of center point is stencil
- 13-point stencil of order 2 for 3-D
# parallel stencil, basic algorithm
- no dependence between output grid points
- boundary cells have boundary conditions that don't get updated
- simple 3-D stencil of order 2, seven points
- each thread does:
	- 13 flops, 7 mul 6 add
	- loads 7 input values
	- 0.46 OP/B
- ratio needs to be larger for the perf to be close to level supported by arith compute
# shared mem tiling for stencil sweep
- almost identical to convolution
- but input tiles don't include corner grid points
- register tiling later!
- for purpose of shared memory tiling: input data reuse in 2D 5-point stencil is much lower than 3x3conv
- each output grid point value uses 5 input grid values and not 9: less reuse
- this discrepancy only grows with dimension!
	- linear vs quadratic in order/radius
- so we'll do thread coarsening and register tiling
- strategies for loading input tiles for conv also apply here
- blocks are same size as input tiles
- turn off some threads for calculating output grid values
- `in_s` in smem to hold input tile for each block
- evaluate effectiveness of shared memory tiling by calculating arith intensity
- larger dimensions
- lots of halo elements
- small tile size also has adverse impact on mem coalescing
# thread coarsening
- overcome block size limitation by coarsening work done by each thread -- one grid point value -> col of grid point values
- each thread loads first layer then second layer
- effectively, can increase tile size without increasing number of threads
# register tiling
- what if only neighbors along x, y, z directions of center point?
- z neighbors for inPrev and inNext can stay in registers
- data reuse spread across registers, not just shared memory
- 