sharded matrices and how to multiply them
# partitioning notation, collective ops
- arrays don't fit in HBM of single device, so have to split them up
- same global/logical shape, but also device local shape
## unified notation for sharding
- variant of named-axis notation to describe how tensor shraded in blocks across devices
- 2D or 3D grid of devices called device mesh
- each axis given mesh names, like X Y Z
- then, describe how each named dim of the array is partitioned across the physical mesh axes
- this is a sharding
- $A[I_X, J_Y]$ means shard first axis $I$ across mesh axis $X$ and second axis $J$ across mesh axis $Y$
- so each shard has $1/(|X| \cdot |Y|)$ of the array
- Mesh: `devices=((0, 1), (2, 3))` and `axis_names=('X', 'Y')`
- $A[I_X, J]$ means $I$ is partitioned across $X$
- $A[I_{XY}, J]$ treats larger axes as a larger flattened dim, partition across all devices; specifies traversal order of partitioning
# computation with sharded arrays
- data distributed across many devices, wish to perform math ops: what are overheads?
- elementwise: no overhead
- ops across elements on many devices? things get complex
- all computation in form of matmuls, simple to analyze
- this is like blockwise matmul!
- then what comms do we add, and how expensive is it?
## case 1: neither multiplicand has sharded contracting dimension
- lemma: when multiplying, computation valid and output follows sharding of inputs unless contracting dimension is sharded or both tensors have a non-contracting dimension sharded along *same* axis
## case 2: one multiplicand has sharded contracting dim
- $\mathbf{A}[I, J_X] \cdot \mathbf{B}[J, K] \rightarrow \mathbf{C}[I, K]$
- can't perform local matmuls of local because we're missing full data from contracting axis of $\mathbf{A}$
- first AllGather shards of $A$ locally, then multiply against $B$
- allgather removes sharding
- $\mathbf{AllGather}_X[I, J_X] \rightarrow A[I, J]$
- this removes a subscript from a set of axes
- how to do an allgather?
- need to pass shards around axis until each device has a copy
- we do ring passes
	- N/2 bidirectional data transfers
	- one direction is $N-1$ hops of size total size/Nper link
	- otherwise ceil(N/2) hops of size 2\*tot_size/N per link
	- how long?
	- $V$ is # bytes
	- $|X|$ is number of shards on contracting dims
	- each hops sends $V/|X|$ in each dir
	- $T_{hop} = \frac{2V}{|X|W_{ICI}}$
	- W_ICI is bidirectional ICI bandwidth
	- need to send $|X|/2$ hops to each TPU
	- $T_{total} = V/W_{ICI}$
	- doesn't depend on $|X|$!
	- even though we have local connections only, locality of connections doesn't matter -- just bottlenecked by speed of each link
- although, we might have min time for a single hop
- for large reds or gathers, bandwidth bound -- so much data that overhead of hop is negligible
## case 3: both multiplicands have sharded contracting dims
- $\mathbf{A}[I, J_X] \cdot \mathbf{B}[J_X, K] \rightarrow C[I, K]$
- local sharded block matrix multiplies are at least possible to perform, since share same sets of contracting indices
- but each product only reps a *partial sum* of full desired product
- $\mathbf{A}[I, J_X] \cdot_{\text{LOCAL}} \mathbf{B}[J_X, K] \rightarrow C[I, K]\{U_X\}$
	- unreduced along X mesh axis, it's incomplete
- need $\mathbf{AllReduce}$
- this removes partial sums, each device along axis has fully-summed value
- allreduce takes array with unreduced axis, sums by passing shards around and accumulating
- AllReduce is twice as expensive as an AllGather
- can see it as a composition of two primitives
- ReduceScatter first, resolve partial sums on array but results in output scattered along given dim
- AllGather unshards the logical axis, collects
- comm time for each hop is per-shared bytes $V$ divided by bandwidth
## case 4: both multiplicands have non-contracting dim sharded along same axis
- sometimes, rule can be violated
- $A[I_X, J] \cdot B[J, K_X] \rightarrow C[I_X, K_X]$
- given shard only has diagonal entry
- can AllGather either A or B
# deeper dive into tpu comm primitives
- AllGather: remove subscript, gather all shards
- ReduceScatter: remove unreduced suffix from array, sum shards across that axis, leave array sharded across second axis
- AllReduce: ReduceScatter+AllGather, removes unreduced suffix and leaves array unsharded along that axis
## all-to-all
- sharded transposition
- rearrange sharded layouts
- move subscript from one axis to another
- doesn't need to replicate all data of each shard across ring, so cheaper by factor of 1/4
## reducescatter
- reducescatter is derivative of allgather
- $\mathbf{AllGather}_X A[I_X] \rightarrow A[I]$
- $\mathbf{ReduceScatter}_X A'[I]\{U_X\} \rightarrow A'[I_X]$
- combine to get sharded derivatives
- likewise,
	- $\mathbf{ReduceScatter}_X(A[I]\{U_X\}) \rightarrow A[I_X]$
	- implies $\mathbf{AllGather}_X(A'[I_X]) \rightarrow A'[I]$
- if we do AllGather and ReduceScatter instead of AllReduce, we can defer AllGather
- we don't want to reassemble the whole matrix product
- ReduceScatter introduces a sharded dim
- so need to specify
# what have we learned?
- sharding specified by mesh
- arithmetic is the same, unless there's a contraction
	- no sharding? no comms
	- one array sharded along contracting dim? allgather one input before op
	- both arrays sharded along contracting dim? local multiply, allreduce or reducescatter
	- both arrays sharded along same mesh axis, non-contracting dim? AllGather one input
- 4 comms primitives
	- AllGather, *unsharding*
	- ReduceScatter, *combining partial results and sharding*
	- AllToAll, *shard axis transposition*
	- AllReduce, *reducescatter+allgather*
- cost is independent of size of axis, only depends on size of input arrays and the bandwidth
- 