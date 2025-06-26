multidimensional grids and data
# multidimensional grid organization
- all threads in grid execute same kernel function
- rely on coordinates/thread indices to distinguish themselves from each other and identify appropriate data to process
- two-level hierarchy: grid of blocks, and block of threads
- all threads in block share same `blockIdx`
- Both levels 3D
- there is also 1D for convenience
- `gridDim` and `blockDim` pre-initialized according to these parameters
- among blocks, `blockIdx.x` ranges from 0 to `gridDim.x-1`
- and so on for y and z
- block organized into 3d array of threads
- 2D blocks from setting blockDim.z to 1, etc
- total size of a block limited to 1024 threads
- grid and blocks do not need same dimensionality
- ordering of block and thread labels is such that highest dimension comes first
# mapping threads to multidimensional data
- thread org usually based on nature of data
- pictures are a 2D array of pixels, so a 2D grid that consists of 2D blocks is convenient
- vertical (row) coordinate: `blockIdx.y * blockDim.y + threadIdx.y`
- horizontal (column) coordinate: `blockIdx.x * blockDim.x + threadIdx.x`
- if extra threads, need if-statement to make sure those threads don't take effect
- how do c statements access elements of dynamically allocated multidimensional arrays?
	- want to access as 2D array
	- row `j`, col `i` is `Pin_d[j][i]`
	- but the standard requires the number of columns in Pin to be known at compile time
	- so 2D arrays need to be linearized
	- this is how everything works, in fact
	- for statically allocated arrays, higher-dim syntax can be used
	- but under the hood the compiler linearizes it to an equivalent 1D array
	- here, the programmer has to do it
	- everything is translated into base + offset
- row-major layour
- there are a total of `blockDim.x * gridDim.x` threads in horizontal direction
- this will generate every integer value from 0 up to this # - 1:
	- `col = blockIdx.x * blockDim.x + threadIdx.x`
- read in r, g, b, write calculated value
# image blur
- what if the threads aren't independent?
- weighted sum of surrounding pixels
- convolution pattern
- here, it's just the average
- 3x3 patch
# matmul
- linear alg functions
	- BLAS: basic linear algebra subprograms
	- level 1: vector ops like $y = \alpha x + y$
	- level 2: matrix-vector ops like $y = \alpha Ax + \beta y$
	- level 3: matrix-matrix like $C = \alpha AB + \beta C$
	- here, $\alpha=1, \beta=0$
- output is a matrix of dot products
- we can make each thread responsible for one element
- can break large matrices into multiple, or calculate more elements per thread
# exercises
1. kernel configs
2. cuda kernel
3. kernel: bd is 16, 32 and gd is (19, 5)
	1. threads per block: 16x32 = 512
	2. number of threads in grid is 95 * 512 = 51200 - 2560 = 48640
	3. blocks in grid: 95
	4. # threads executing: 150 x 300 = 45000
4. 2d matrix, width 400, height 500. row 20, col 10
	1. row-major: 20 x 400 + 10 = 8010
	2. column-major: 10 x 500 + 20 = 5020
5. width 400, height 500, depth 300. index of x=10, y=20, z=5?
	1. $W(Hy + z) + x$