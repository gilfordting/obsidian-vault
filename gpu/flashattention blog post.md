website features
- [ ] running list of definitions
- [ ] footnotes/sidebar
- [ ] table of contents
- [ ] visualizations
	- [ ] if gif: slider
	- [ ] switch between 2D and 3D, for cuda block grid
- [ ] pseudocode vs cuda for kernels (avoid annoying things like array indexing)
- [ ] code w copying supported, syntax highlighting, different text, language support
- [ ] collapse menu for things like definitions/conventions
- [ ] graph dependencies for blog post sequence
- [ ] feedback mechanism - highlight and send suggestion? text box at bottom?

# Kernel 1: No parallelism
```python
N = 1024
d = 128
Q, K, V, O = [] # Nxd matrices
S, P = [] # NxN matrices

# Calculate S, pre-softmax matrix
for i in range(N):
	for j in range(N):
		val = 0.0
		for k in range(d):
		# we are indexing in to K^T, so we swap the indices
		# this is equivalent to declaring K_prime = K^T
		# and doing Q[i][k] * K_prime[k][j],
		# like a normal dot-product structure
			val += Q[i][k] * K[j][k]
		S[i][j] = val

# Calculate P, post-softmax matrix
for i in range(N):
	# Apply softmax on row S[i]:
	row = S[i]
	# Calculate the maximum element
	max_val = -float('inf')
	for j in range(N):
		max_val = max(max_val, row[j])
	# Calculate the sumexp of this row
	sum_exp = 0.0
	for j in range(N):
		sum_exp += exp(row[j] - max_val)
	# Then we calculate a row of P
	P_row = P[i]
	for j in range(N):
		P_row[j] = exp(row[j]) / sum_exp

# Last step: matrix-multiply P and V to get O
for i in range(N):
	for j in range(d):
		# For each element of the Nxd output, we do a dot-product
		value = 0.0
		for k in range(N):
			value += P[i][k] * V[k][j]
		O[i][j] = value
```

The first step of introducing parallelism into this kernel is going to utilize the CUDA thread group hierarchy. This is a crucial component of the CUDA programming model

Generally, when we parallelize by splitting a computation into blocks, this looks like the *removal* of one loop, and the usage of a variable called `threadIdx`. Thus, figuring out how to break your larger problem into smaller chunks is the first step of parallelizing the task.


rough outline:
- kernel-host interaction

# Arithmetic intensity and the roofline model
Knowing how to *analyze* any machine learning workload is a crucial step of the optimization process - there are some things you can easily deduce without even writing any code, or doing any profiling.

For as much hype as it gets, machine learning basically just boils down into 3 steps:
1. Load in a bunch of numbers from memory. (These are your model weights, as well as your input data.)
2. Do a bunch of math on those numbers.
3. Take the result of all that math and store it somewhere in memory.

Notice there's two fundamentally different things that are going here. Steps 1 and 3 are all about data *movement* -- we need to move those numbers from main memory, where they're accessible by only the CPU, to GPU memory [^data-movement-required], so we can actually do stuff with them. Then we move our computed result from GPU memory back to main memory, so we can do things like display LLM output text to the user. Conversely, Step 2 is all about *computation* -- modern machine learning is mostly just lots of matrix multiplications, over and over, which itself consists of various multiplications and additions on floating-point numbers.

These two classes of work correspond to different hardware units as well -- there are physical units of circuitry that move data around, as well as compute units designed to do floating-point operations. Both of these units have pretty impressive capabilities -- on an NVIDIA A100 GPU, memory bandwidth goes up to 1,935GB/s, and FP32 throughput goes up to 19.5 TFLOPS (Tera FLoating-point OPerations per Second). And in an ideal world, you'd be hitting these numbers all the time -- if you've paid the premium for one of NVIDIA's fancy GPUs, you want to milk them for all they're worth! __segue__

Let's simplify our model even further. We can view a **throughput-oriented processor** like a GPU[^throughput-processor] as a set of two distinct units through which data can flow: a memory unit with a max flow rate limited by memory bandwidth, and a compute unit with a max flow rate limited by compute throughput. Fundamentally, our machine learning program describes how we push data through this system. Say we have a processor with a 100B/s memory bandwidth and 400 FLOPS/s compute throughput:
**diagram here**

Now consider what happens within the span of one second: the compute unit does at most 400 FLOPS, and at most 100 bytes of data passing through the memory unit. (For our simplified model, we don't make a distinction between writes/inputs or reads/outputs; we just look at how much data the memory unit touches.) Clearly, the best scenario is when we're running both units at maximum speed -- for every 100 bytes of data loaded or stored, we perform 400 FLOPS. The *ratio* between compute and memory is what we care about, since it will generalize to any arbitrary time period, not just 1-second intervals. In performance engineering, this ratio is known as *arithmetic intensity* -- formally, it's defined as the number of FLOPs executed per byte transferred by the memory unit (again, including both loads and stores). So for this simple processor, the arithmetic intensity of the ideal program is 4 FLOPS/B.

But we won't always get the best-case scenario -- not all programs have the same arithmetic intensity. Clearly, lots of programs will under-utilize at least one of the two units. Consider the following cases:
- We're running the memory unit at max speed, and 100 bytes of data pass through every second. But the compute unit is only at half capacity, and is doing 200 FLOPS per second. This is a program with an AI of 2 FLOPS/B, which is **less** than the ideal. If we could just feed the compute unit with more data, we could squeeze more juice out of it, but our memory unit can't go any faster. This is a **memory-bound** regime.
- The compute unit is going as fast as it can, crunching numbers and doing 400 FLOPS every second. Meanwhile, the memory unit is barely being used, loading/storing 25 bytes per second. This is a program with AI = 16 FLOPS/B, which is **more** than the ideal. Our memory unit could definitely pump out more data per second, but it wouldn't get consumed because our compute is at max capacity. This is a **compute-bound** regime.

So if our program's AI is too low, it's memory-bound. If it's too high, it's compute-bound. The ideal balance is somewhere in the middle.

It turns out that with *large* language models, though, a naive implementation is severely memory-bound. Memory operations take a really long time, compared to things like multiplication and addition; loading from main memory takes around ~200 cycles, but a multiplication is only 5 cycles. So we need to be very clever with reusing data as much as possible



- We'd get more juice out of the compute unit if were able to get more data
- 
- We could 
- We'd like to feed it more data -- it's certainly able to handle 
- 
- But what if our program has an arithmetic intensity of 2 FLOPS/B

Let's think about what happens when we have an AI that's below or above 4 FLOPS/B:
- < 4 FLOPS/B: 

We might underutilize the memory unit, or the compute unit, or both. Consider what happens in these scenarios:
- 



This is a characteristic of the given workload -- in other words, the arithmetic intensity is within your control.

way you write your program determines 



If we want to maximize our hardware usage, what does this imply about the most ideal program?

The key is to consider the *ratio* between the number of FLOPS done by the program and the amount of data it reads from or writes to main memory. For this processor, if we can 






To understand why we can't get peak performance straight out of the box, let's consider a few different workloads:

i think better examples are needed
- **High memory, low compute.**  Imagine we load every single number from our GPU's memory into registers -- but then we do absolutely nothing with them. Maybe we'll add a few of them together. A GPU with 80 GB of HBM contains 80 GB / 4 B = 20 *billion* 32-bit floating-point numbers. Even if we load everything as fast as possible at peak memory bandwidth, we're doing so little computation that there's no way we're maxing out our compute units -- they're mostly just waiting around for new data so they can do a tiny amount of math. It's like being stuck in traffic; it doesn't matter how fast your car can go if the cars in front of you are barely moving.
- **Low memory, high compute.** Let's say we had the opposite workload -- we loaded a *single* number from memory and did a bunch of computation on it. (Something like starting with 0 and adding 1 until it reaches $2^{31}-1$, the maximum possible integer.) Same deal here -- the compute units are all active and doing as much work as they can, but they're just not getting very much data.

So whether we're compute or memory-bound depends on how the workload is balanced between the two. To measure this, the typical metric we use is *arithmetic intensity*. Simply put, the arithmetic intensity is the answer to the question: when we load in our data, how much useful work are we doing on it before we write the final result back to main memory? The units of arithmetic intensity are typically FLOPS/B.

Our first workload is **low arithmetic intensity** -- we don't do much computation per-byte. Our second workload the opposite, **high arithmetic intensity** -- we're doing a comically large amount of computation per byte. In the former case, we were limited by memory bandwidth; in the latter, we were limited by compute throughput. So at some point, 

It's important to note that arithmetic intensity is simply an *approximation* of what performance we'll be able to get.

-- data *movement*

One metric that we'll look at a lot is the *arithmetic intensity* of the task at hand.

 This can't be the amount of data

^data-movement-required: It's possible, I suppose, that you could 
^throughput-processor: latency, not throughput