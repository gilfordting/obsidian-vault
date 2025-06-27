memory architecture and data locality
# importance of mem access efficiency
- let's calculate the expected performance level of the most executed portion of matmul -- the dot product
- compute to global mem access ratio -- number of FLOPs for each byte access of global memory within region
- arithmetic intensity
- A100:
	- low arithmetic intensity means you're not using enough compute, because it takes so long to load data
	- there are also tensor cores -- which make this disparity even higher
- roofline model
	- x-axis is arithmetic intensity
	- y-axis is throughput
	- horizontal line is peak computation
	- positive slope line is peak memory bandwidth
	- all points must be *under* the curve
	- we want to get as close to the curve as possible
- so, higher perf is dependent on arithmetic intensity
- reduce number of global memory accesses
- need intrinsic data reuse
# cuda mem types
- global memory and constant memory
	- former is R/W, latter is read-only but short-latency
- local memory, also R/W
	- actually placed in global memory; not shared across threads
	- each thread has own section of global mem
	- used for statically allcoated arrays, spilled registers, other elements of call stack, etc
- registers and shared memory are on-chip memories
	- these are much faster
	- registers allocated to individual threads
	- shared memory is allocated to thread *blocks*
	- all threads in block can access
- cpu vs gpu register architecture
	- zero-overhead scheduling because registers of incoming already in register file
	- need larger register file
	- GPUs also have dynamic resource partitioning
	- CPU register architecture dedicates fixed set of registers per thread regardless of thread's actual demand
- register file is on processor chip, so short access latency and higher bandwidth compared to global mem
- access to registers also involves fewer instructions
	- arithmetic operations are often on registers by default
	- if a value is in global mem, need to `load` before `fadd`, for example
- also, energy reasons
	- register << global mem for energy efficiency
	- occupancy that is achieved for application can be reduced, if reg usage in full-occupancy scenarios exceeds limit
	- avoid oversubscribing to limited resource when possible
- shared memory is also memory load operation
- variables in shared mem accessible by all threads in block
- high-bandwidth sharing of data among threads
- simultaneous data access
- registers, local mem, shared mem, global mem have different functionalities, latencies, bandwidth
- cuda -> hardware arch. mapping
	- automatic variables: registers, thread-scope
	- arrays: local, thread-scope
	- `__shared__`: shared, block
	- global, constant: what you expect
- variables that are not arrays are *scalar*
- scalar vars declared in kernel, device functions are put in registers
- automatic array vars not stored in registers
- stored in local memory
- not typically used, though
- shared vars indicated by `__shared__`
- scope is thread block
- `__constant__` outside of any function scope
- scope is all grids
- kept in global mem but cached for efficient access
- global vars are slow
- so cross-block collaboration is possible
- but needs atomics
# tiling for reduced mem traffic
- partition data into subsets called tiles, these fit in shared mem
- computation must be independent
# tiled matmul kernel
- scope of shared mem is block
- advance across inner dimension by tile granularity
- first barrier: all threads finish loading tile
- second barrier: make sure all elements are used from shared memory, before proceeding to next iteration -- this way no threads load stuff that corrupts input values of other threads
- first is read-after-write dependence
- second is write-after-read
- former is true dependence -- really needs the data, so has to wait
- write-after-read is false dependence: writing thread doesn't need anything from the reading thread
	- the same memory location is being reused; will not exist if diff locations
- this is strip-mining
- creates phases
- tiling is also used for CPUs, but there is implicit reliance on the cache
# boundary checks
- arbitrary-width matrices
- need to make sure indices are in bounds of array being accessed
# impact of memory usage on occupancy
- shared memory is a resource
- we can dynamically decide using `cudaGetDeviceProperties`
- C `extern` keyword for dynamically allocated arrays
- sizes passed as arguments
# exercises
1. shared memory does not help; there is no reuse
2. baseline: each of the 64 output elements requires 16 global mem reads; 1024 accesses
	1. 2x2 tiling: 16 supergrids, and each one requires 32 global mem reads; 512 accesses
	2. 4x4 tiling: 4 supergrids, and each one requires 64 global mem reads; 256 accesses
3. read stale value when computing dot-product, or read future value
4. communication within a block, since registers are private. also lower latency
5. 32x improvement
6. 512,000
7. 1000
8. two square $N \times N$ matrices, how many accesses?
	1. no tiling: $2N^3$, each of the $N^2$ output elements reads $2N$ elements from global mem
	2. $T \times T$ times: supergrid of size $\frac{N}{T} \times \frac{N}{T}$, each subgrid reads $2TN$ elements from global mem, overall $2N^3/T$ global mem accesses
9. 36 flops / 28 B = 1.29 AI
	1. bandwidth-bound, since mem bandwidth * AI < peak FLOPS
	2. compute-bound, since mem bw * AI > peak FLOPS
10. tiling
	1. only blocksize 1 works
	2. sync barriers
11. kernel: 128 tpb, 8 blocks
	1. 1024
	2. 1024
	3. 8
	4. 8
	5. one 4-byte scalar, 128-long 4-byte array so 129 x 4 = 516
	6. global mem: 4x2 from first loop, 1 on line 12, 5 on line 14 -- 14x4B = 56 B. 4 additions, 5 muls = 9 fp ops
12. 2048 threads/SM, 32 blocks/SM, 65,536 registers/SM, 96kb smem/SM
	1. 65 TPB, 27 RPT, 4KB shared mem. 64x27=1728 RPB
		1. 31.5 blocks/SM from thread restrictions
		2. 37.9 blocks/SM from reg restrictions
		3. 24 blocks from smem restrictions