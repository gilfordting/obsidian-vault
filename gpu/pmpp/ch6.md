perf considerations
# memory coalescing
- global mem is bandwidth
- DRAM, capacitors
- to make data access rate go up, just use parallelism
- why is DRAM so slow?
	- can take a while to charge/discharge line
	- takes time to raise the voltage level
- so we request a range of consecutive locations
- these are DRAM bursts, so if we make focused use of data, it's good
- so mem access needs to be favorable pattern
- use the fact that threads in a warp exec same instruction at any given pt in time
- if accesses are far away, hardware cannot coalesce
- we can rearrange thread mapping, or rearrange data layout
- transfer global --> shared in coalesced manner
	- corner turning
- this is like carpooling
# hiding memory latency
- 