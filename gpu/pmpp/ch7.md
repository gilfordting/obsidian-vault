intro to constant memory and caching
# background
- array op: output data element is weighted sum of input element and collection of input elements surrounding it
- filter array/convolution kernel
- referred to as convolution filters
- size of filter is $2r+1$ and $r$ is radius -- # elements on either side
- can be thought of as inner product
- boundary conditions?
- assign a default value, like 0
- "ghost cell"
# parallel convolution
- we can calculate all output elements in parallel
- mapping threads to output elements?
- threads in 2D grid, have each thread calculate one output element
- each thread block is 4x4, grid is 4x4
- doubly nested loop
- register value `Pvalue` accumulates intermediate results, save DRAM bandwidth
- since 0 values for ghost cells, just skip multiplication
- two observations
	- control flow divergence, only happens for boundary conditions, which is small portion of output
	- memory bandwidth, low arithmetic intensity
# constant memory and caching
- filter array $F$ has 3 interesting properties
	- usually small, $r \leq 7$
	- constant, do not change
	- all threads access this
	- and all access in same order!
- this is an excellent candidate for constant memory and caching
- we can declare variables to reside in constant memory, with `__constant__` keyword
- small constant memory, only modifiable by host
- `cudaMemcpyToSymbol(dest, src, size)`
	- `dest` points to location in constant memory
	- `src` is pointer to source data in host memory
	- `size` is # of bytes to be copied
- kernel functions access constant mem variables as global
- constant memory variables in DRAM, but because we know no modifications we can aggressively cache -- this is what hardware does
- DRAM forms bottleneck, so we usually mitigate with on-chip cache memories -- reduce # variables that need to be accessed from main memory
- caches are transparent -- not directly controlled
- if we wanted to use CUDA shared memory to hold global value, declare var as `__shared__`, explicitly copy
- with caches, program tries to access original variable and the hardware maps original address to a location in cache
- tradeoff between size of memory and speed of memory
- L1 cache is directly attached to processor core, but small
- L2 large but slower
	- shared among processor cores, access bandwidth shared among SMs
- do not need to support writes by threads
- constant memory is small so a cache can be effective, constant cache
- effectively, all $F$ elements always accessed from constant cache
- assume no DRAM bandwidth on access to $F$
- the input elements can also benefit from caching!
# tiled convolution with halo cells
- threads collaborate to load input elements into on-chip memory for subsequent use
- collection of output elements processed by each block is output tile
- 16 blocks of 16 threads each
	- in practice, at least one warp per block (and ideally more)
- assume $F$ elements in constant mem
- input tile is the input elements needed to calculate the elements in an output tile
- dimensions need to be extended by radius of filter, $r$
- in practice, output tile dimensions much larger and input tile size / output tile size ratio is close to 1.0
- how to organize threads to address this discrepancy?
- launch thread blocks with dimension matching *input* tiles
- some of the threads need to be disabled during calculation of output elements
- this can reduce efficiency
- second approach: launch blocks with dimension that of output tiles
- input tile loading is more complex, but output elements are simpler to calculate
- all threads perform barrier sync to ensure entire input tile loaded
- blocks that handle tiles at edge: threads that handle ghost cells do not perform mem accesses
- but for large input arrays, the effect is insignificant
```python
# Pseudocode for CUDA kernel
IN_TILE_DIM = 32
OUT_TILE_DIM = IN_TILE_DIM - 2 * FILTER_RADIUS

# Launched as an T_i x T_i thread block -- also a (T_o + 2r) x (T_o + 2r) block
def convolution(N, P, width, height):
	# calculate position within output tile, offset by r to collaboratively load
	col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS
	row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS
	# declare shared mem for tile
	N_shared = init_array(IN_TILE_DIM, IN_TILE_DIM)
	if in_bounds(row, col):
		N_shared[threadIdx.y][threadIdx.x] = N[row][col]
	else:
		N_shared[threadIdx.y][threadIdx.x] = 0.0
	__syncthreads() # barrier sync
	# calculate output elements next!

	# use these to index into shared memory
	tileCol = threadIdx.x - FILTER_RADIUS
	tileRow = threadIdx.y - FILTER_RADIUS

	# disable threads at the edge of the block
	if in_bounds(row, col): # compare against width and height
		if in_bounds(tileCol, tileRow): # compare against OUT_TILE_DIM
			value = 0.0
			for filterRow in range(2*FILTER_RADIUS+1):
				for filterCol in range(2*FILTER_RADIUS+1):
					value += F[filterRow][filterCol] * N_shared[tileRow+filterRow][tileCol+filterCol]
			P[row, col] = value
```

```c
#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
__constant__ float F_c[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];
__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P, int width, int height) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;

    // loading input tile
    __shared__ N_s[IN_TILE_DIM][IN_TILE_DIM];
    if(row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0;
    }

    __syncthreads();

    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    // turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >=0 && row < height) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM && tileRow>=0 
            && tileRow<OUT_TILE_DIM){
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += F[fRow][fCol]*N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width+col] = Pvalue;
        }
    }
}
```
- arithmetic intensity?
- let $r$ be filter radius, and $T_i$ be in-tile dimension and $T_o$ be out-tile dimension, where $T_o=T_i + 2r$
- each thread assigned to output tile element performs one multiplication and one addition for every element
	- $T_o^2 \times (2r+1)^2 \times 2$ ops
		- $T_o^2$ output elements
		- each output element comes from convolving with $(2r+1)^2$-size filter
		- and 2 ops per filter element
	- $4T_i^2 = 4(T_o+2r)^2$ mem accesses
		- 4 bytes per element of input tile
	- then overall ratio is $\frac{T_o^2 \times (2r+1)^2 \times 2}{4(T_o+2r)^2}$
	- simplifying, we can assume $T_o \gg 2r$ so we have $4T_o^2$ below
	- so this becomes $\frac{(2r+1)^2}{2}$ ratio
	- we get better arith intensity with larger filters!
	- but we're limited to $32\times 32$ limit for thread block size
- larger filter size has higher ratio, because each input element is used by more threads
- larger filter size has higher disparity between bound and ratio actually achieved -- larger number of halo elements that force a smaller output tile
- small tile sizes, because insufficient on-chip memory
	- especially for 3d convolution!
# tiled convolution using caches for halo cells
- lots of complexity, because input tiles and blocks are larger than output cells
- *halo cells* are also the internal elements of neighboring tiles!
- there's a chance that by the time a block needs halo cells, already in L2 cache bc of neighboring blocks
- so naturally served from L2 cache, without DRAM traffic!
- so leave accesses to halo cells in original N elements, instead of loading into shared memory
- then shared memory needs to hold only internal elements of tile
- new kernel
	- input size = output size
	- loading shared mem becomes simpler
	- halo cells are not loaded, so no danger of loading ghost cells
	- just usual boundary condition
	- inner loop is more complicated: either load from shared memory or regular memory
- subtle advantage: block size, input tile size, output tile size can be the same and a power of 2
# summary
- general pattern that forms basis of parallel algs
- can view stencil as special case of convolution
# exercises
1. [10, 15, 4] * [8, 2, 5] = 80 + 30 + 20 = 130
2. grunt work
3. filters
	1. identity
	2. shift left
	3. shift right
	4. weighted difference, edge detection
	5. blurring
4. 1D conv, array of size $N$, filter of size $M$ (assuming that $M = 2r+1$)
	1. $2r$
	2. $N(2r+1) = NM$
	3. left side: leftmost used once, second-leftmost used twice, so $1 + 2 + \cdots + r=r(r+1)/2$ and doing this on both sides means $NM - r^2-r$ multiplications
5. similar
6. similar
7. annoying futher
