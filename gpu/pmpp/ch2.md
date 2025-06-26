heterogeneous data parallel computing
# data parallelism
- problem is usually that there is a lot of data to process
- but each individual data point can be processed independently
- there's also task parallelism, which is exposed through task decomposition of apps
- vector addition and matrix-vector multiply
- can the tasks be done independently?
# cuda c program structure
- there is a host (CPU) and devices (GPUs)
- exec starts with host code
- when kernel function called, large number of threads launched on device to execute kernel
- thread is a simplified view of how a processor executes sequential program
- consists of:
	- code of program
	- point in code being executed (PC)
	- vals of variables, data structures
- thread execution is effectively sequential
# vector addition kernel
- function that mutates via pointer
- how to parallelize? move calculations to device
	- allocate mem for a, b, c
	- copy a and b to device mem
	- call kernel to do computation
	- copy c from device mem
	- free device vectors
- two API functions for allocating and freeing device global mem
	- `cudaMalloc`: allocate piece of device global mem for object
		- address of pointer variable that should be set to point to allocated object
		- should be cast to (void \*\*) because function expects generic pointer
		- malloc is generic function, not restricted to any particular type of objects
		- this way, `cudaMalloc` can write the address of the allocated memory into provided pointer var, regardless of type
		- host code that calls kernels passes this pointer value to the kernels that need to access the mem object
		- second param gives size of the data to be allocated, in # bytes -- this is consistent with size param to C malloc fcn
		- address passed in is mutated
	- `cudaFree` called with same pointer, to free storage space for A vector from device's global mem
		- this value is not modified, only read in order to return allocated mem back to available pool
		- value, not address
- addresses should not be dereferenced in host code
	- this can cause exceptions, accessing memory not assigned to you technically
- once host code has allocated space in device global mem, request that data transferred from host to device
- accomplished by calling one of the cuda api functions
- `cudaMemcpy`, four params
	- pointer to destination location
	- pointer to source location
	- number of bytes to copy
	- types of memory involved: H/D->H/D
- `vecAdd` calls cudaMemcpy to copy from host to device
- same function can be used to transfer data in both directions -- order the source and dest, and use appropriate constant for transfer type
- error checking is also a thing
# kernel functions and threading
- kernel func specified code to be executed by all threads during parallel phase
- all threads execute same code, so SPMD (single prog multiple data)
- cuda runtime will launch a grid of threads that are organized into two-level hierarchy
- grid is an array of thread blocks
	- blocks all same size, can contain up to 1024 threads
- total number of threads in each thread block is specified by host code, when kernel called
- same kernel can be called w different \#s of threads at diff parts of host code
- for a given grid, number of threads in block in `blockDim`
	- 3-D, x, y, z
	- can use some prefix of these
- number of threads in each dim of thread block should be multiple of 32, for hardware efficiency reasons (warp size?)
- two more built-ins, `threadIdx` and `blockIdx`
- this allows threads to distinguish themselves from each other; determines area of data each thread is to work on
- hierarchical organization allows for locality
- `blockIdx` var gives all threads in block common block coordinate
- we can combine `threadIdx` and `blockIdx` to create unique global index
- so with 1-D threadIdx and blockIdx, can make unique global index
	- here, calc as `i = blockIdx.x * blockDim.x + threadIdx.x`
	- `blockDim` is 256, so $i$ values in block 0 range from 0 to 255, then block 1 is 256 to 511, and so on
	- so coverage is continuous
- kernels don't have access to host memory
- `__global__` keyword indicates that function is kernel
- for dynamic parallelism, can also be called from deviceheterogeneous data parallel computing
- calling the kernel results in new grid of threads being launched
- `__device__` indicates that function being declared is a cuda device function
	- executes on cuda device, can be called only from kernel function or other device function
	- executed by the thread that calls it
	- does not result in any new device threads
	- recursion and indirect (helper?) function calls are possible
	- but this should be avoided, for maximal portability?
- `__host__` means function being declared is cuda host function
	- executes on host, can only be called from host
	- no keyword? host by default
- can use both `__host__` and `__device__` in func declaration
	- compiler generates two versions of object code
	- lots of library functions fall in this category
- other extensions: `threadIdx`, `blockIdx`, `blockDim`
- all threads execute same kernel code! so there needs to be some way to distinguish themselves from each other
- these allow threads to access the hardware registers that provide these identifying coordinates
- local vars get their own value, and aren't visible to other threads
- a loop is removed?
	- now replaced with grid of threads
	- sometimes called loop parallelism
- `if (i < n)`: not all vector lengths can be expressed as multiples of block size
# calling kernel functions
- having implemented kernel, remaining step is to call that function from host code to launch grid
- grid and thread block dimensions set via execution config parameters
- these are given between `<<<` and `>>>`
- first is # of blocks in grid
- second is # of threads in each block
- `<<<ceil(n, 256.0), 256>>>`
- code is hardwired to use thread blocks of 256 threads each
- but the number of thread blocks is variable
- thread blocks can be executed in any arbitrary order
- and they can also be executed in parallel
- this gives kernels scalability in execution speed with hardware
# compilation
- implementing cuda c kernels requires using various extensions -- these aren't part of C
- so we can't just use a C compiler
- NVCC (nvidia c compiler) does this
- it produces host code and device code
	- host code is ANSI C, can compile with standard C/C++ and run as trad CPU process
	- device code, marked with CUDA keywords that designate CUDA kernels and assoc. helper functions and data structures, compiled by NVCC into PTX
- PTX compiled by runtime component of NVCC into real object files, executed on CUDA-capable GPU
# exercises
1. C
2. C
3. D, because different blocks process twice the number of elements
4. C, 8 thread blocks
5. D
6. D
7. C (dest, source, num, constant)
8. C
9. kernel
	1. 128 threads
	2. 1563 blocks x 128 tpb = 200,064
	3. 1563 blocks
	4. 200,064
	5. 200,000
10. use both keywords