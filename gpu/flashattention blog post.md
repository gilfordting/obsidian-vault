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