[arxiv](https://arxiv.org/abs/2307.08691)
# abstract
- suboptimal work partitioning between different thread blocks and warps
- low-occupancy or unnecessary shared memory reads/writes
- better work partitioning now:
	- reduce number of non-matmul FLOPs
	- parallelize attention computation across different thread blocks
	- within each thread block, distribute work between warps to reduce comm through shared memory
- yield about 2x speedup
# 1 intro
- suboptimal work partitioning
- 3.1: reduce number of non-matmul FLOPs
- parallelize along sequence length dimension, not just batch and # heads
# 2 background
## 2.1 hardware characteristics
- perf characteristics, execution model
- threads within warp can communicate by fast shuffle, or cooperate using matrix multiply
## 2.2 std attention impl
- S for QKT, P for softmax, O = PV
- for MHA, computation performed in parallel across many heads, parallel over batch dim
## 2.3 flashattention
### 2.3.1 forward pass
- tiling and online softmax with rescaling of output
### 2.3.2 backwards
- recompute S and P, avoid storing them
# 3 flashattention-2: algorithm, parallelism, work partitioning
- parallelize between thread blocks
- also, partition work between warps to reduce shared mem access
## 3.1 algorithm
- reduce number of non-matmul FLOPs
### 3.1.1 forward pass
- don't have to rescale both, just maintain unscaled version of $O$ and keep around statistics
- don't save both max and sum of exponentials, just logsumexp
- causal masking
	- if col indices > row indices, skip computation of block
	- for each row, only apply causal mask to one block
### 3.1.2 backward pass
- also just use row-wise logsumexp
## 3.2 parallelism
- FA1: parallelize over batch size, number of heads
- 1 thread block per attention head
- number of thread blocks: batch size x number of heads
- each thread block runs on an SM, an A100 for example has 108 SMs
- scheduling efficient when number is large
- long sequences (small batch sizes, small number of heads) means we also parallelize over sequence length dimension
- outer loop over sequence length is embarassingly parallel
- swapping order of loop (outer: row blocks, inner: column blocks)
- softmax dependence is between columns
- parallelize across Q
# 3.3 work partitioning between warps
- within a thread block, we still have to partition between different warps
- 4-8 warps per thread block, usually
- FA would split K and V across 4 warps, and Q is accessible by all warps
- FA2 splits Q across 4 warps, and K/V is accessible by all
- 