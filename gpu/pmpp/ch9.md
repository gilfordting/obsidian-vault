parallel histogram
- each output can be updated by any thread
- coordination among threads is needed
- baseline uses atomics
- next, privatization
# background
- histogram is display of number count, percentage of occurrences of data values in dataset
- can be computed sequentially
- intervals of 4
- complexity is $O(N)$
- data array elements accessed sequentially in for-loop, using cache lines well
- memory-bound
# atomic ops and basic histogram kernel
- launch as many threads as data, have each thread process one input element
- read and increment counter
- output interference
- this is a read-modify-write update on a memory location
- there are race conditions
- we fix this with atomics
	- realized with hardware support for locking
- atomics don't enforce any particular exec order
- `atomicAdd` is intrinsic, compiled into hardware atomic op instruction
- intrinsic functions
	- modern processors offer special instructions -- perform critical functionality like atomics, or perf like vector ops
	- exposed as intrinstic functions, as part of a library
	- compilers see this and translate directly -- these are not asm function calls, just inline special instructions
- new kernel, cuda-style:
	- calculate `i` from `threadIdx`, `blockIdx`, etc
	- increment is now `atomicAdd`
# latency and throughput of atomic operations
- serializing simultaneous updates
- serializing any portion of massively parallel program can increase execution time, reduce execution speed
- access latency to DRAM can be hundreds of clock cycles
- this breaks down for many atomic ops
- duration of atomic op is latency of memory load + latency of memory store
- this limits throughput
- how to improve throughput?
- reduce access latency to heavily contended locations
- cache memories used for this
- atomics can be performed in last-level cache, shared by all SMs
# privatization
- direct traffic away from heavily contended locations
- replicate contended data structs into private copies
- can increase throughput
- but need to be merged
- common approach: private copy for each thread block
- contention only within the same block, and when merging at the end
- host allocates enough memory for private copies
- reduce level of contention by factor approx # active blocks
- at end of execution, thread blocks commit values in private copy into version produced by block 0
- one benefit is that we can use `__syncthreads()` to wait for each other before committing
- if private copy accessed by multiple blocks, need to call another kernel to merge private copies
- any reduction in latency directly translates into improved throughput of atomic ops on same mem location
- we can put data in shared memory
- barrier sync to ensure that all bins of private histo properly initialized
- atomic add on shared memory
# coarsening
- overhead of privatization: need to commit private copies to public copy
- the more thread blocks we use, the larger the overhead
- reduce number of committed private copies by reducing # blocks and having each thread process multiple inputs
- a single thread should not access contiguous elements
- rather, a group of threads should request contiguous elements at roughly the same time
# aggregation
- aggregate updates into a single one
- so if a thread encounters a run of same values, can coalesce
- if no contention/heavy contention, little control div -- either all threads be flushing or all in streak
- control divergence compensated by reduced contention
# summary
- output location data-dependent -- can't apply owner-computes rule