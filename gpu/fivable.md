[link](https://library.fiveable.me/parallel-and-distributed-computing/unit-12/cuda-kernel-optimization-techniques/study-guide/rxIvWYwl0ITaHYOP)
# optimizing cuda kernels
- thread coarsening: work of multiple threads into one, reduce overhead
- loop unrolling: reduce branch penalties, increase ILP
- memory coalescing to use memory bus effectively
- occupancy opt: adjust resource usage
- instruction-level opt:
	- use intrinsic functions for fast math ops
	- avoid thread div within warp
	- utilize fast math when precision allows
- profiling tools
	- nvidia visual profiler
	- nvidia nsight compute
# minimizing cpu-gpu comms
- use cuda streams to overlap computation with data movement
	- async data transfers
- pinned (page-locked) mem alloc improves transfer speeds, preventing mem from being swapped to disk
- unified memory: single memspace accessible by both CPU/GPU
- use `cudaMallocManaged()`, CUDA runtime will automatically migrate data
- kernel fusion
- zero-copy mem allows GPU to access host mem beneficial for access patterns
	- read from RAM directly, sent over PCIe
# shared mem for data reuse
- tiling
- size is limited, balanced against number of thread blocks
- bank conflicts limit performance
- ensure threads access diff banks
- shared mem alloc dynamic -- size can be set at kernel launch time
# efficient parallel reduction and scan ops
- parallel reduction is fundamental
- sequential addressing reduces bank conflicts, improves mem coalescing
- loop unrolling reduces # iters, increases ILP
- warp-level primitives, `__shfl_down_sync()` -- no smem usage
- hierarchical approach: block-level reduction followed by global reduction
	- atomics
- parallel scan (prefix sum) achieve $O(n)$ work and $O(\log n)$ step complexity