[link](https://modal.com/gpu-glossary)
# hardware
## cuda
- no more fixed pipeline
- uniform hardware units
## streaming multiprocessor (SM)
- programming GPUs = producing sequences of instructions for SMs to carry out
- roughly analogous to cores of CPUs
- exec computation, store state in registers
- execution is pipelined within an instruction
- no specexec, no PC prediction
- can execute more threads in parallel, at the same time and on the same clock cycle
- also support for *concurrent* threads -- interleaving instructions of threads of execution
- cpus can also run threads concurrently
	- but switches between warps happen at the speed of a *single* clock cycle, which is >1000x faster than cpu
	- powered by warp schedulers
- so lots of available warps + speed of warp switches hide latency caused by memory reads, thread sync, other expensive instructions
- this ensures compute resources are well-utilized
## core
- make up an SM
- cuda cores, tensor cores
- gpu cores kind of like cpu, component that effects actual computations
- more like a pipe in which data goes in, transformed data is returned
- these pipes are associated with specific instructions
- different fundamental affordances of throughput from programmer's
- SMs are closer to being equivalent of cpu cores
- they have register memory, cores to transform that info, and instruction scheduler
## special function unit (SFU)
- SFUs accelerate certain arithmetic ops
- `exp`, `sin`, `cos` etc
## load/store unit (LSU)
- dispatch requests to load/store data to memory subsystems
- they interact with SM's on-chip SRAM and off-chip global RAM
- lowest and highest levels of mem hierarchy
## warp scheduler
- this decides which threads to execute
- warps are switched out on a per-clock basis
- cpu context switches are super expensive because need to save context of one thread and restore another
- reduced locality from context switches
- but on gpu, each thread has own private registers allocated from register file -- no data movement needed
- L1 caches can be entirely programmer managed, and are shared between the warps scheduled together on an SM
## cuda core
- gpu cores, execute scalar arithmetic instructions
- not generally independently scheduled
- *groups* of cores are issued the same instruction by warp scheduler
- applied to different registers
## tensor core
- operate on more data for a single instruction
- example: `HMMA16.16816.F32` SASS instruction
	- calculate D = AB + C
	- MMA is matrix multiple and accumulate
	- HMMA16: inputs are half-precision 16 bit
	- outputs are accumulated into 32-bit single-precision floats
	- 16, 8, 16 are dimensions of matrices
	- m, k, n for mxk @ kxn
- but we don't just put this instruction in a single thread
- we have to put it across the entire warp
- most of the power overhead is in decoding, shared across a warp thanks to warp scheduler
- kind of like CISC hardware
- assembler-level instruction might be produced by compiler to implement PTX-level matrix multiple accumulate instructions, like `wmma`
- these instructions are compiled into many SASS instructions on smaller matrices
- max performance: need to use PTX intrinsics, not pure CUDA
- tensor cores are much larger and less numerous than cuda cores, one per warp scheduler
## streaming multiprocessor architecture
- versioned with particular arch that determines compatibility with SASS
- target versions can be specified when invoking `nvcc`, cuda compiler
## texture processing cluster (TPC)
- not encountered in contemp discussions of gpus
- not mapped onto level of mem hierarchy or thready hierarchy
## graphics/gpu processing cluster (GPC)
- TPCs plus raster engine
- additional layer of thread hierarchy
## register file
- stores bits in between manipulation by cores
- split into 32-bit registers that can be dynamicaly reallocated between datatypes
	- 32-bit ints
	- 64-bit floats
	- two 16-bit floats at a time
- allocation of registers to threads done by `nvcc`, optimizing register usage with thread blocks
## l1 data cache
- private SM memory
- each SM partitions the mem among groups of threads scheduled onto it
- co-located with, nearly as fast as components effecting computations
- SRAM
- accessed by load/store units, LSUs
- gpus: cache mostly programmer-managed
## gpu ram
- global memory of ram is large memory store that is addressable by all SMs
- also GPU RAM or video RAM, VRAM
- uses DRAM cells, slower but smaller
- not on same die as SMs, located on shared interposer for decreased latency and increased bandwidth
- this is for global memory, and spilled registers
# software
## cuda programming model
- three key abstractions
	- hierarchy of thread groups: threads -> blocks -> grids
	- memory hierarchy; lowest layer should be nearly as fast as an instruction exec
	- barrier sync: coordinate within thread group w barriers
- this way, express programs that scale transparently
- prevents gpus from failing to get faster under new hardware
- example:
	- each thread block can coordinate tightly within the block, but can't communicate across blocks
	- so the blocks capture parallelizable components of the program, can be scheduled in any order
	- the programmer "exposes" this parallelism to the compiler and hardware
	- when there are more SMs, more blocks can be executed in parallel
	- so it naturally gets faster with better architecture
- cuda C -> PTX -> SASS
## streaming assembler (SASS)
- assembly format for programs running on GPUs
- lowest-level format in which human-readable code can be written
- output by `nvcc` alongside PTX
- streaming in streaming assembler refers to SMs
- versioned and tied to specific architecture
- some examples for hopper GPUs
	- `FFMA R0, R7, R0, 1.5`: fused floating point multiply add: R0 = R7 * R0 + 1.5
	- `S2UR UR4, SR_CTAID.X`: copy X value of cooperative thread array's index from special register to uniform register 4
- this is very uncommon to write by hand
## parallel thread execution (PTX)
- IR for code that will run on parallel processor
- output by `nvcc`
- both virtual machine and instruction set architecture
- in PTX, program runs with same semantics on multiple machines
- kind of like x86_64, aarch64, SPARC -- but very much an IR like LLVM-IR
- these components of CUDA binary JIT compiled by host CUDA drivers into device-specific SASS
- for NVIDIA GPUs, PTX forward-compatible
- examples
	- `.reg .f32 %f<7>;` -- compiler directive for PTX-to-SASS, indicating that the kernel consumes seven 32-bit floating point registers
		- registers dynamically allocated to groups of threads from SM's register file
	- `fma.rn.f32 %f5, %f4, %f3, 0f3FC00000;` -- apply fused multiply-add to do: f5 = f3 * f4 + 0f3FC00000
		- rn suffix sets floating point rounding mode to round even
```
mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ntid.x
mov.u32 %r3, %tid.x
```
- move x-axis values of cooperative thread array index, cooperative thread array dimension index, and thread index into three u32 registers r1-r3
- ptx programming model exposes multiple levels of parallelism to programmer
- these levels map directly onto hardware through PTX machine model
- single instruction unit for multiple processors
- each processor runs one thread
- processor runs one thread each, threads execute same instructions, so parallel thread execution
- coordinate with shared memory, private registers to capture different results
- inline PTX uncommon but not unheard of
- important for performance
- this is the only way to take advantage of some specific features like `wgmma` and `tma` on hopper
## compute capability
- instructions in PTX ISA compatible with only some GPUs
- so this is used to abstract away details of physical GPUs
- onion layer model, only add new stuff without changing old
## thread
- thread of execution
- lowest unit of programming, atom of the thread group hierarchy
- thread has own registers, little else
- both sass and ptx program target threads
- thread can have private instruction pointer/PC
- but usually written such that all threads in warp have same instruction pointer, execute in lockstep
- also have stacks in global mem. for storing spilled registers, function call stack
- single core executes instructions from single thread
## warp
- group of threads that are scheduled together, exec in parallel
- all threads in warp scheduled on single SM
- a single SM executes multiple warps
- or at the very least all warps from same thread block/cooperative thread array
- unit of execution
- all threads of warp execute same instruction, SIMT model
- when warp issued instruction, results not available within single clock cycle, usually -- can't issue dependent instruction
	- example: global memory
	- but also some intense arithmetic instructions
- instead of waiting for warp to return results, when multiple warps scheduled? just run another warp
- latency-hiding for throughput
- maximize number of warps scheduled onto each SM
- not actually part of hierarchy: implementation detail
- somewhat akin to cache lines -- don't directly control, and don't need to consider for *correctness* but matter for perf
## cooperative thread array (CTA)
- collection of threads scheduled onto same SM
- PTX/SASS implementation of thread blocks
- one or more warps
- can direct threads within CTA to coordinate with each other
- shared memory in L1 data cache
- threads in diff CTAs can't coordinate with each other via barriers, like threads within a CTA can
- must coordinate via global memory
- CTA exec order is indeterminate
- number of CTAs scheduled onto single SM depends on some factors
- limited set of resources: lines in RF, slots for warps, bytes of SMEM
- these resources are calculated at compile time
## kernel
- unit of CUDA code that programmers write and compose
- kernel is launched once and returns once
- but it's executed many times, once each by number of threads
- executions are concurrent and parallel
- collection of all threads executing kernel organized as kernel grid, thread block grid
- kernel grid executes across multiple SMs, operates at scale of entire GPU
- matching level of mem hierarchy is global mem
- kernels are passed pointers to global memory on device when invoked by host, and they just mutate memory
- two kernels
	- first one: each thread computes *one* element of output matrix
		- each thread does one FLOP per read from GMEM, bad throughput since cuda core bandwidth higher than memory
	- second one: tiled
## thread block
- level of hierarchy below a grid, but above a warp
- equivalent of cooperative thread array
- blocks are the smallest unit of thread coordination exposed to programmers
- must execute independently, so any execution order is valid -- from all permutations of fully serial to all interleavings
- single cuda kernel produces one or more thread blocks
- each of these contains one or more warps
- blocks are usually multiples of warp size
## thread block grid
- can be one, two, or three dimensions
- made of thread blocks
- matching level is global mem
- thread blocks are independent units of computation; range from fully sequential to fully in parallel
## memory hierarchy
- matches thread group hierarchy
- managed by programmer
- thread block grid: global mem, access coordinated with atomics and barriers
- single thread: chunk of register file, private memory
- shared memory: thread block level, L1
- careful management of cache is why kernels are hard
## registers
- store info manipulated by single thread
- stored in register file of SM, can also spill to global memory
- not directly manipulated by CUDA
- visible to PTX and SASS, managed by `nvcc`
## smem
- level of mem hierarchy corresponding to thread block level
- kernel is:
	- global mem -> shared mem
	- perform ops on those data
	- optional: synchronize threads within thread block
	- write data back to global mem
		- prevent races across thread blocks with atomics
## gmem
- every thread can access for the lifetime of program
- atomics always
- within CTA, can more tightly sync with barriers
- allocated from host using memory allocator provided by CUDA driver API or CUDA runtime API
# host
## cuda software platform
- collection of software for developing cuda programs
## cuda c++
- implementation of cuda programming language, extension of c++
- features:
	- kernel def with `global`, C functions taking in pointers and having return type `void`
	- kernel launches with `<<<>>>`, these set thread block grid dimensions
	- shared memory alloc with `shared` keyword, barrier sync with `__syncthreads()` intrinsic, thread block and thread indexing with `blockDim` and `threadIdx` built-ins
- these are compiled with `gcc` and `nvcc`
## gpu drivers
- mediate interaction between host programs/OS and GPU
- primary interfaces: CUDA runtime API and CUDA driver API
## `nvidia.ko`
- binary kernel module file at core of nvidia gpu drivers
- executes in privileged mode, comm. directly with hardware on behalf of user
## cuda driver api
- userspace component of nvidia cuda drivers
- stuff like `cuMalloc`
- few cuda programs written to directly use this api, instead use cuda runtime api
- not linked statically, linked dynamically
- binary-compatible; app compiled against old versions of cuda driver api can run on systems with newer versions of cuda driver api
## `libcuda.so`
- typical name for binary shared object file that implements cuda driver api
## nvidia management library (NVML)
- monitor and manage state of nvidia gpus
- expose power draw and temp of GPU, allocated mem, device power limit and power limiting state
## `libnvml.so`
- typical name for binary shared obj file that impl features of NVML on linux
## `nvidia-smi`
- query and manage state of gpu
- also list processes using gpu
## cuda runtime api
- wraps driver api
## `libcudart.so`
- typical name for binary shared obj file that impl cuda runtime api on linux
## nvidia cuda compiler driver (nvcc)
- outputs binaries conforming to ABI, includes PTX/SASS
## nvidia runtime compiler (nvrtc)
- compile cuda c++ to ptx without requiring launch of nvidia cuda compiler driver
## nvidia cuda profiling tools interface (CUPTI)
- sync timestamps across cpu host, gpu device
## nvidia nsight systems
- perf debugging tool for c++ programs
- 