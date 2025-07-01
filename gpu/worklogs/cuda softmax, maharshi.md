[link](https://maharshi.bearblog.dev/optimizing-softmax-cuda/)
# math
- input vector $x \in \mathbb{R}^N$ produces output vector $O \in \mathbb{R}^N$
	- $O_i = \frac{e^{x_i}}{\sum_{k=0}^N e^{x_k}}$
- softmax op depends on current element $x_i$ and sum of exponentials of all elements of input vector $x$ -- this sum is the normalization factor, or norm
- usually instead of single vector, we deal with matrix of shape $(M, N)$: $M$ rows, where each row is a vector of $N$ elements. softmax along columns
- output is matrix of same shape
- example of vector with 5 elements
```python
import torch
import torch.nn.functional as F

vector = torch.randn(5, dtype=torch.float32)
print("Input vector:", vector)

# softmax along the last dimension
output = F.softmax(vector, dim=-1)
print("Output vector:", output)
```

```python
Input vector: tensor([-1.3701,  0.7485,  0.1610, -2.0154,  1.0918])

Output vector: tensor([0.0382, 0.3176, 0.1765, 0.0200, 0.4477])
```

- values of $x_i$ being very large or small: exponentials might cause overflow or underflow considering precision limits of floating point numbers
- not numerically stable
- subtract max value $x_m$ from each $x_i$ before computing exponential
- this shifts the numbers to a range that can work with FP numbers
- equivalent to $O'_i = e^{-x_m}O_i$
# how fast is pytorch?
- 7.226 ms
# kernel 1, naive softmax
- each thread in block processes and computes one entire row of input matrix
- three passes:
	- calculate maximum
	- calculate norm
	- calculate softmax
# kernel 2, online softmax
- fuse the first pass with calculating the norm?
- at each step, we multiply current norm with correction term
- improvement
# kernel 3, shared memory and reductions
- have each block process a row, and threads within block process a chunk of the entire row
- memory coalescing, consecutive addresses from GMEM is faster than accessing random addresses
- we need reductions to calculate max
- each thread has private set of variables called `local_max` and `local_norm`, `N_THREADS` threads in total
- reduction is done with $O(\log N)$ time complexity
# kernel 4, shuffle instructions
- warps directly communicate
- so reduce to first element of warp, then warp leaders will communicate with shared memory