[link](https://developer.download.nvidia.com/GTC/PDF/1083_Wang.pdf)
software-hardware correspondence
- thread : cuda core
- thread block : SM
- grid : device
- several concurrent TBs on one SM
- blocks divide in groups of 32, basic scheduling units
# memory optimization
- coalescing is the single most important consideration
- requests from warp falling in L1 cache line
- caching load: warp requests 32 aligned, consecutive 4-byte words
	- warp needs 128 bytes, utilization 100%
	- same if permuted
- if fall within 2 cache lines: utilization 50%
- bank conflicts
- shared mem divided into banks
	- number of banks is 32, and each bank assigned to 4-byte word
- if two R/W fall in same bank? access serialized
- special cases: all access same word, continuous byte/double
- strive for perfect coalescing
# latency optimization
- mem/instruction TP far from peak
- latency hiding: context switching
- increase concurrency, adjust resource usage to increase active warps (TLP)
- number of blocks >> # of SM > 100 to scale well to future device
- block size multiple of 32
- ratio of active warps per SM to max number of allowed warps: occupancy
- occupancy must be high
- resources:
	- registers
	- shared memory
	- thread block slots
	- thread slots
- any of these can be limiting factor
- if adding single instruction leads to significant perf drop, occupancy
- required occupancy: depends on both arch and application
- increase ILP of each thread
# instruction optimization
- reduce instruction count, use high throughput instructions
- avoid automatic conversion of double to float (add 'f' to floating literals, the default is double)
- fast math: `func()` vs `__func()`; `-use-fast-math` forces the latter
- control flow, divergent branches
- avoid diverging *within* a warp
# cpu-gpu interaction optimization
- overlapped execution using streams
- host <-> device data transfer much lower BW than global memory
- minimize transfer: intermediate data directly on GPU
- one large transfer much better than many small ones
- streams and async api
- default api: kernel launches async with cpu
- memcopies (D2H, H2D) block CPU thread
- driver serializes CUDA calls
- streams and async functions: stream is seq of ops executing in issue-order
- overlap kernel and memory copy