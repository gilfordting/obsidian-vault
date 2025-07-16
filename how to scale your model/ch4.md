transformer math
# counting dots
- dot product requires $P$ adds and multiplies, $2P$ flops
- matrix-vector does $2NP$
- matmul does $2NPM$
- two higher-dim arrays C and D, some dims are contracting and some batching, flops is product of all dims where batch/contract only counded once
- batching must be in both
- compute scales cubically, data transfer scales quadratically
- with larger models, easier to hit compute-saturated limit
## forward and reverse flops
- more flops during backprop
- $B$ is matrix in network
- $A$ is input activations
- $C = AB$
- $A$ is $N \times P$
- $B$ is $P \times M$
- then derivative of loss wrt B given by chain rule
- $\frac{\partial L}{\partial B} = \frac{\partial L}{\partial C} \frac{\partial C}{\partial B} = A^T\left(\frac{\partial L}{\partial C}\right)$
	- 2NPM
- $PM$ is size of params
- so 2 flops per token
- then same for backwards, but two of these because need to compute gradient for both params and activations
# global flops and params calculation
## mlps
- 2 input matmuls, element-wise gating, output matmul
- 18BTDF FLOPs, 3DF params
- dot-product attention op
- dot-product attention FLOPs only become dominant during training once $T > 8D$
# misc math
## MoEs
- dynamic routing of MLPs
- two comms, two AllToAlls that route tokens and bring them back home
## gradient checkpointing
- we need to save everything from fwd pass if we want to do backprop, if we want to avoid recomputing
## kv caching
- prefill and generation
- 