roofline analysis
# where does the time go?
- deep learning model is just a bunch of matmuls, which is a bunch of flops
- T_math = Computation FLOPs/Accelerator FLOPs/s
- so we need time for compute
- also, we need communication *within* a chip
- transfer tensor between on-chip memory (HBM) and compute cores
- this speed is the HBM bandwidth
- also, communication *between* chips
- this is when you have multiple accelerators
- T_comms = Comm Bytes / Network or Mem Bandwidth Bytes/s
- we can have computation overlapped with communication
- so lower bound is everything in parallel: $T_{\text{lower}} = \max(T_{\text{math}}, T_{\text{comms}})$
- upper bound is fully sequential: $T_{\text{upper}} = T_{\text{math}} + T_{\text{comms}}$
- sum $\leq$ 2 x max
- what if we get perfect overlap?
- T_math > T_comms: compute-bound
- T_comms > T_math: communication bound
- how to tell which one? arithmetic intensity
- ratio of total FLOPs it performs to number of bytes it needs to communicate
- either within chip or between chips
- FLOPs per byte of given operation
- high arith intensity: use most of available FLOPs
- Intensity(Computation) > Intensity(Accelerator)
## visualizing rooflines
- use roofline plot, peak flops of alg on hardware against arith intensity
- improve perf by increasing arith intensity or by increasing mem bandwidth available
## matmul
- $X \times Y \rightarrow Z$
- $X$ is $B\times D$, $Y$ is $D \times F$, $Z$ is $B \times F$
- bf16 precision
- mem: load $2BD$ bytes for $X$, $2DF$ bytes for $Y$, write $2BF$ bytes for $Z$
- compute: $BF$ entries, each one is a dot product of approx $2D$, so $2DBF$ FLOPs
- intensity is $\frac{2BDF}{2BD+2DF+2BF}$, can cancel the 2
- nice simplification if local batch size $B$ is small relative to $D$ and $F$, this becomes $\frac{BDF}{DF} \approx B$
- reasonable for transformer matmuls, models have batch size in tokens $B < 1024$ but $D$ and $F$ > 8000
- compute-bound when local batch size greater than 240 tokens
## network communication rooflines
- comms between chips matter, matrices across multiple TPUs
- what if we have matrices split evenly across $D$ dimension on 2 T/GPUs
- then we can do a matmul on each, and copy partial sums to a single TPU to add
- each TPU does half the work, so we effectively get twice the compute bandwidth
	- we also need to accumulate $BF$ partial sums but that's negligible
- T_comms is now communication time between chips, total bytes send divided by network bandwidth