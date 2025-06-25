# heterogeneous parallel computing
- two main trajectories
	- multicore, maintain execution speed of sequential programs while moving into multiple cores
	- many-thread, focusing on execution throughput of // apps
- why is there a large peak performance gap?
	- cpu optimized for sequential code, minimize effective latency
	- caches, branch prediction, control logic
	- gpu is throughput-oriented
	- must be able to move large amounts of data quickly
	- gpus are at an advantage here
- reducing latency more expensive than increasing throughput in terms of power and chip area
- so lots of threads, but each one can have longer latency
- throughput-oriented design
# why more speed or parallelism?
- main motivation: apps continue to get faster with better hardware
- 