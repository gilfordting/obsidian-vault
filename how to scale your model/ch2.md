how to think about TPUs
# what is a TPU?
- compute core specializing in matmul
- attached to stack of fast mem, HBM
- HBM has weights, activations, optimizer states, new batch data
- bandwidth determines how fast data transfers between HBM and compute
- Smem and Vmem, scalar and vector
- scalar is like CPU, dispatches instructions to VPU and MXU
- VPU performs elementwise ops, activations and such, and loads data into MXU
- MXU performs matmuls, is driver of chip FLOP/s
- tensorcore is basically really good matmul machine
- MXU
	- core of tensorcore
	- one bf16 8x128 @ bf16 128x128 -> f23 8x128 matmul
	- every 8 cycles
	- systolic array
	- also lower precision matmuls with higher TP
	- Appendix B
- VPU
	- math ops like ReLU, or pointwise add/mul
	- reducations also here
	- appendix C
- VMEM
	- on-chip scratchpad, near compute
	- smaller than HBM, but has higher BW to MXU
	- like an L1/L2 cache
- very fast at matmul
- HBM
	- big chunk at fast memory that stores tensors for use by tensorcore
	- usual computation is HBM -> VMEM -> MXU -> VMEM -> HBM
	- HBM and TensorCore bandwidth limits computation in memory-bound workloads
- all ops pipelined and overlapped
- for matmul, copy chunks of matrices A and X
- matmul is pipelined, so copies are overlapped with work
- if load from HBM to VMEM is slower than FLOPs in MXU or Vector Unit, bandwidth bound since starving VPU or MXU of work
- VMEM and arith intensity: VMEM smaller than HBM but higher bandwidth to MXU
- fit weights in VMEM instead of HBM? matmuls FLOPs bound at smaller batch sizes
- lower arith intensity can still be efficient, but VMEM so small this often a challenge
- TPU chip is two TPU cores which share memory and can be thought of as one accelerator with twice the FLOPs, megacore configuration
- chips are in set of 4 on a tray connected to CPU host via PCIe
- PCIe bandwidth limited
# TPU networking
- chips connected to each other through ICI network in a Pod
- 4 nearest neighbors connected, with edge links to form 2D torus
- 6 neighbors for 3D torus in v4, v5
- these are direct links
- toroidal structure reduces max distance between any two nodes from $N$ to $N/2$
- also twisted torus config, wraps torus in Mobius-strip like topology
- reduce avg distance between nodes
- TPU pods can get big
- nearest-neighbor connectivity a key diff between TPUs and GPUs
- GPUs connected with hierarchy of switches that approx point-to-point connection between every GPU
- rather than local connections like TPU
- GPUs within a node are directly connected; larger topologies need $O(\log N)$ hops between each GPU
- but this also means that GPUs can send arbitrary data within a node in single low-latency hop
- TPUs are dramatically cheaper, since NVLink switches are expensive, and simpler to wire together
- larger topologies, because # links per device and bandwidth per device is constant
- ICI fast relative to DCN, slower than HBM bandwidth still
- so when we split models across multiple chips, need to avoid bottlenecking MXU with slower cross-device comms
- set of ICI connected TPUs is called a slice
- different slices can be connected with DCN
# key takeaways
- hierarchy: tpu connected to memory -> other chips over ICI -> rest of datacenter over DCN
- within slice, TPUs only connected to nearest neighbors via ICI
- weight matrices need to be padded to 128
- lower precision matmuls are faster
# appendix
# a: gpus
### overview
- gpus have simpler communication model, more complicated programming model
- GPUs have more SMs (analogous to TensorCore)
- more flexible computation but more complex reasoning
- cuda cores do SIMD scalar work, L1 cache used to speed data access and for register spilling
### networking
- nvidia GPUs in cliques of 8-256 GPUs via switches
- point-to-point comms in a clique
- but between cliques is a lot slower
- so training on more than 256 requires pipe parallel to scale, which is more complex
- PaLM trained on two cliques of 3072 TPU chips, by contrast
- for neural net ops like AllReduce, all-to-all conns don't have an advantage -- same comms patterns, regardless
- but can store MoE models across more GPUs, transmit experts around more efficiently
## b: systolic array
- 2D grid of ALUs, each can do multiply and add
- weights passed from above, inputs passed from left
- weights are partially loaded first, diagonally, and activations fed in, also diagonally
- multiply all overlapped green and blue units, sum result with any residual from above, and pass result in turn down one unit
- can efficiently pipeline this to multiply large matrices
## c: tpu internals
### scalar core
- processes all instructions and executes transfers from HBM into VMEM
- scalar core responsible for fetching instructions for VPU, MXU, XLU
### vpu
- 2d vector arith unit of shape 8, 128
- 