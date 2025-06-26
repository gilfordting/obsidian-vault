compute arch and scheduling
# architecture of modern gpu
- organized into array of highly threaded streaming multiprocessors, SMs
- each SM has several processing units called streaming processors or CUDA cores
- small tiles inside SMs
- these SMs also have on-chip memory structures
- and there is also GBs of off-chip memory
- older GPUs used double data rate synchronous DRAM
- but newer ones use high-bandwidth memory, DRAM modules tightly integrated with GPU
# block scheduling
- when kernel called, cuda runtime launches grid of threads that execute kernel code
- multiple blocks are likely to be simultaneously assigned to same SM
- but a limited number of them can be simultaneously assigned to a given SM
- variety of factors depend
- so there's a limit on how many blocks can be executing in a CUDA device
- threads in the same block are scheduled simultaneously on the same SM, because the blocks are scheduled all-or-nothing
- so threads across different blocks need to work harder to communicate
- within the same block, we get:
	- barrier synchronization
	- and shared memory
- threads in same block can coordinate activities using barrier sync function `__syncthreads()`
- when a thread calls this, it blocks until *every* thread in the same block reaches there too
- if a `__syncthreads()` is present, it must be executed by all threads in block
- when `__syncthreads()` placed in if-statement, either all threads execute block that includes it, or none does
- needs to be all or nothing for branching
- placing them in diff branches means they are different barrier sync points
- using barrier sync improperly can result in deadlock
- execute in close time proximity, to avoid excessively long waiting times
- a thread that never arrives at barrier sync can cause deadlock
- the block can only begin once it has all the resources needed for all of the threads
- blocks can be executed in any order
# warps and simd hardware
- blocks can execute in any order relative to each other
- but what about the execution timing of threads within each block?
- assume threads in a block can execute in any order wrt to each other
- algos with phases: barrier sync used before going to the next phase
- thread scheduling is a *hardware implementation* concept
- once a block assigned to SM, further divided into 32-thread units called *warps*
- these are the unit of thread scheduling
- the blocks are divided into warps
- warps are formed of threads with consecutive `threadIdx` values
- 0-31 first warp, 32-63 second warp, etc
- calc # of warps in SM for given block size, and given number of blocks assigned to each SM
- if each block has 256 threads, each block must have 8 warps
- 3 blocks in SM => 24 warps
- blocks are partitioned into warps on the basis of thread indices
- 1-D is easy
- in general, warp $n$ starts with thread $32n$ and ends with $32(n+1)-1$
- if not multiple of 3, last warp padded with inactive threads
- 48-thread block --> fully-active warp, half-active warp (padded w 16 inactive)
- blocks consisting of multiple dimensions of threads: dimensions projected into linearized row-major layout before partitioning into warps
- larger $y$ and $z$ coords come after those with lower ones
- placed with increasing `threadIdx.x` values
- an SM executes all threads in warp using SIMD model
- a single instruction is fetched, and given to all the threads in the warp
- cores in SM are grouped into processing blocks -- every 8 cores form processing block, and they share an instruction fetch/dispatch unit
- A100:
	- 64 cores
	- organized into 4 processing blocks, 16 cores each
	- threads in the same warp assigned to same processing block
	- fetches instructions for the warp, executes for all threads in warp at the same time
- all threads in a warp need to execute the same instruction at any point in time
- sometimes called single instruction, multiple thread
- what's the advantage of simd?
	- the control hardware is shared across execution units, so smaller % of hardware for that and larger % for arithmetic throughput
- warps and simd hardware
	- von neumann model
	- i/o to provide programs and data
	- first, program and data goes in memory
	- control unit has program counter, PC: mem address for next instruction
	- PC used to fetch instruction into IR
	- instruction bits used to see what to do
	- this model is also called the stored program model -- change behavior of computer, by storing diff program in memory
	- what's this like as a gpu?
	- same, one control unit
	- but the control signals now go to multiple processing units!
	- these each correspond to a core in the SM
	- and each of these executes one of the threads
	- so how do they differ?
	- because they have different data operand values in register files
	- an instruction like `add r1, r2, r3` will have diff values of `r2` and `r3` across different processing units
	- control in modern processors is complex
# control divergence
- things work well when all threads follow same exec path
- but if there are different control flow paths?
- we take multiple passes through these paths, one pass for each
- in an if-else construct, the hardware takes two passes
- turn one set of threads on for if, turn the complement on for else
- so if there are different execution paths, the threads exhibit control divergence
- this way, we can get the full semantics of cuda threads
- cost of divergence is the extra passes the hardware takes in order to allow diff threads to make own decisions
- these passes might be executed concurrently!
- independent thread scheduling
- also, control-flow
- what about a for-loop?
- if the decision condition is based on threadIdx values, can cause thread divergence
- so why use control constructs?
- one thing is boundary conditions
- the size of the data can be an arbitrary number
- the performance impact of control divergence decreases as size increases
	- why??
# warp scheduling, latency tolerance
- when threads assigned to SMs, more threads assigned than cores in SM
- each SM has only enough execution units to execute a *subset* of all the threads
- so why is this?
- has to do with toleration of long-latency ops like memory access
- when a warp needs to wait, we switch out to another resident warp
- this is latency tolerance, or latency hiding:
	- like when you don't want someone to hold up the line by filling a form, so you have them step aside and help the next person
- this is also used for other things, like pipelined FP arithmetic and branch instructions
- enough warps means the hardware will find a warp to execute at any point in time
- make full use of execution hardware, while instructions of some warps wait for the results of these long-latency ops
- there is no wasted time: zero-overhead thread scheduling
- threads, context-switching, zero-overhead scheduling
	- thread, again, is code, instruction pointer, state of data
	- von Neumann mode: code in memory, PC is address of instruction, IR holds instruction, register and memory hold values
	- modern processors allow context-switching, with time-sharing
	- saving and restoring PC value and contents of registers and memory means we can suspend execution of thread, correctly resume execution of thread later
	- saving and restoring register contents during context-switching can incur overhead
	- zero-overhead scheduling: silencing a warp and waking up another one doesn't introduce idle cycles
	- trad cpus can't do this -- they need to save register contents, load execution state, etc
	- all execution state is in hardware registers -- if we assign different registers to diff threads, they will naturally avoid conflict
- oversubscribing of threads to SMs is essential for latency tolerance
# resource partitioning and occupancy
- not always possible to assign SM the max number it supports
- ratio of actual / max is occupancy
- how are SM resources partitioned?
- the resources are registers, shared memory, thread block slots, thread slots
- ampere A100 can support max of 32 blocks/SM, 64 warps (2048 threads) /SM, 1024 threads/block
- we can partition thread slots among blocks
- so SMs can either execute many blocks, each with a few threads
- or they can execute few blocks, each with many threads
- so each block doesn't always need the same amount of resources
- this avoids waste, and generalizes
- there are some interactions that cause underutilization
- what about when each block has 32 threads?
	- there are 2048 thread slots, and they need to be assigned to 64 blocks
	- but the volta SM can only support 32 block slots at once
	- so then only 1024 thread slots (32 blocks x 32 tpb), which is 50% occupancy
	- for max occupancy, we need 64 tpb
- also bad if max number of TPB not div by block size
	- up to 2048 threads per SM supported
	- if block size is 768, SM can only accommodate 2 thread blocks and 512 thread slots are not utilized
	- 1536/2048 = 75% usage
- this is even without registers and shared memory
- automatic/local variables declared in cuda kernel are put in registers
- some kernels require more regs per thread than others
- so there is dynamism here too
- but it's limited -- max of 65,536 regs per SM
- full occupancy? each SM needs enough registers for 2048 threads; this is 32 registers per thread max
- in some cases, register spilling happens
- this comes at the cost of higher read latency
- consider a kernel with 31 regs/thread, 512 threads/block
	- 2048 threads / (512 tpb) = 4 blocks simultaneously
	- 2048 threads * 31 regs/thread = 63,488 regs
	- 2 more local vars: register limit now exceeded, and we only assign 3 blocks to each SM
	- this lowers our occupancy
# querying device properties
- how do we know these properties of the device?
- amount of resources specified as part of compute capability
- cuda runtime has api function `cudaGetDeviceCount`
- lots of PCs come with integrated GPUs
- also `cudaGetDeviceProperties`
- and a built-in type `cudaDeviceProp` has fields representing properties of CUDA device
- properties
	- `maxThreadsPerBlock`
	- `multiProcessorCount`, # of SMs
	- `clockRate`, freq
	- `maxThreadsDim[0], [1], [2]`: max number of threads allowed along each dim of block
	- `regsPerBlock`: # of registers available in each SM
	- `warpSize`
# exercises
1. kernel
	1. 128 threads, 4 warps
	2. 8 blocks, 32 warps
	3. statement
		1. consider 4 warps at a time, and then multiply by 8 -- fully inactive, 24/32, 32, 8/32. 24 active
		2. 16 divergent
		3. 0%
		4. 3/4
		5. 1/4
	4. statement
		1. all active, 32
		2. all divergent, 32
		3. 50%
	5. ranges from j < 5 to j < 3
		1. 3 divergence-free iterations
		2. 2 diverging
2. 2048
3. 2 warps? or just 1, because the last one is useless
4. $1 - \frac{1}{n * t_{max}}\sum^n t_i$ = 1 - 1/(8\*2.9)\*(2+2.3+3+2.8+2.4+1.9+2.6+2.9) = 14%
5. no, because threads can still take a diff amount of time? or also this decreases throughput?
6. c, 3 blocks
7. occupancy
	1. possible, 50%
	2. possible, 50%
	3. possible, 50%
	4. possible, 100%
	5. possible, 100%
8. limiting factors
	1. 128 tpb, 30 rpt, 3840 rpb; 16 blocks from threads, 17.07 blocks from registers, full occupancy
	2. 32 tpb, 29 rpt, 928 rpb; 64 blocks from threads but 32 max; 70.62 blocks from registers; 50%
	3. 256 tpb, 34 rpt, 8704 rpb; 8 blocks from threads, 7 blocks from registers, 7/8 occupancy
9. slow?