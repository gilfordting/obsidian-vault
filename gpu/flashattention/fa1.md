[arxiv](https://arxiv.org/abs/2205.14135)
# abstract
- IO-aware exact attention using tiling to reduce # of mem r/w between HBM and SRAM
- also block-sparse attention for approximate algorithm
# 1 intro
- transformers are dominant
- long context is hard
- self-attention is $O(N^2)$
- so make that faster!
- lots of existing methods but no wall-clock speedup because focused on FLOP reduction
- compute speed outpaces memory speed -- bottleneck is memory
- IO-aware algorithms critical for memory-bound ops
- examples: database joins, image processing, numerical linear algebra, and more
- avoid reading and writing attention matrix to/from HBM
	- compute softmax without access to entire input
	- don't store large intermediate attention matrix for backwards pass
- techniques used
	- tiling
	- store softmax norm factor from forward pass, in order to quickly recompute attention on-chip -- better than reading intermediate attention matrix from HBM
- IO complexity is $O(N^2d^2/M)$ ($d$ is head dim, $M$ is SRAM size)
- standard attention: $\Omega(Nd+N^2)$
- for usual $d$ and $M$, many times fewer HBM access
- block-sparse flashattention as well
# 2 background
## 2.1 hardware perf
- gpu mem hierarchy
- execution model
- performance characteristics, compute vs memory bound
- kernel fusion
- but intermediate values still written to HBM :(
## 2.2 standard attention implementation
- intermediates for $S$, pre-softmax matrix, and $P$, post-softmax and pre-multiplication matrix
# 3 flashattention algorithm, analysis, and extensions
- forward pass only (appendix B for backwards)
## 3.1 efficient alg with tiling, recomputation
- reduce amt of HBM accesses to sub-quadratic in $N$
- attention is computed by blocks
- since softmax is over rows, softmax couples columns of $K$
- we can compute softmax one block at a time!
- $l$ for exp-sum, $m$ for max
- $O_i$ will be written to multiple times
	- each time, the previous value is corrected
- typical values of $d$ and $M$ means $d^2$ much smaller than $M$
- given SRAM size of $M$, blocks of $K$ and $V$ of size $\Theta(M)$ can be loaded
- for each block of $K$ and $V$, iterate over all blocks of $Q$