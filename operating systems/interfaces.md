- what's the job of an operating system?
	- share a computer among multiple programs
	- provide a more useful set of services than just what the hardware supports
	- manages and abstracts the low-level hardware
	- shares the hardware across multiple programs, so they can appear to run at the same time
	- provide controlled ways for programs to interact, so they can share data or work together
- services are provided to user programs through interface
- design is tricky to get right!
	- would like it to be simple and narrow, to get impl. right
	- but we might also want to provide lots of sophisticated features
	- how do we do this? a few mechanisms that can be combined to provide generality
- uses `xv6`, for concrete examples
	- basic interfaces introduced by Unix; also mimics Unix's internal design
	- Unix has narrow interface with mechanisms that combine well -- surprising degree of generality
	- lots of modern OSes have Unix-like interfaces
- `xv6` takes traditional form of a *kernel*, special program that provides services to running programs
	- each running program (process) has memory with instructions, data, and stack
		- instructions: implement computation
		- data: variables on which computation acts
		- stack: organizes program's procedure calls
	- lots of processes, but only one kernel
- what if the process needs to invoke a kernel service? invoke a *system call*
	- this is something provided by the OS interface
	- the syscall enters the kernel, kernel performs service, and return
	- so alternation between user space and kernel space
- kernel uses hardware protection mechanisms provided by CPU, ensures that each process executing in user space can access only own memory
	- so the kernel has hardware privileges required to implement these protections
	- user programs do not have these privileges
	- so syscalls raise the privilege level and executes a *pre-arranged* function in the kernel
- collection of system calls provided by kernel: the interface that user programs see
- `xv6` kernel provides subset of Unix ones
- shell is used for examples
	- it's an ordinary program that takes user commands and executes them
	- it's not part of the kernel
	- there's nothing special about it -- this is the power of the syscall interface
# processes and memory
- `xv6` process consists of user-space memory (instructions, data, stack) and per-process state private to the kernel
- processes are time-shared
- registers are saved upon context switch
- kernel associates process identifier, `pid`, with each process
- can create new process using `fork` syscall
	- copies over memory, instructions, data, and stack
	- returns in both original and new processes
	- in original process, `fork` returns new process's PID
	- in new process, returns 0
	- parent and child
![[Pasted image 20250612223613.png]]
(all syscalls)
- example fragment:
```c
int pid = fork();
if (pid > 0) {
	printf("parent: child=%d\n", pid);
	pid = wait((int *) 0);
	printf("child %d is done\n", pid);
} else if (pid == 0) {
	printf("child: exiting\n");
	exit(0);
} else {
	printf("fork error\n");
}
```
- `exit` causes calling process to stop exec, release resources like memory, open files
	- 0 for success, 1 for failure (integer status argument)
	- `wait` returns PID of exited (or killed) child of current process, copies exit status of child to passed address
	- if none of caller's children exited, `wait` waits for one to do so
	- but if there are no children, `wait` returns -1
	- don't care about exit status? pass 0 address
- these processes *do not share memory* - value of `pid` in child stays the same
- `exec` syscall replaces calling process's memory with new mem image, loaded from file stored in filesystem
- file needs a particular format: specifies which parts are instructions vs data, which instruction to start at, etc
	- ELF format used here
	- result of compiling source code
- when `exec` succeeds, does not return to calling program
	- instead, instructions loaded start from entry point as declared
	- two args: name of file with executable, array of string args
- example:
```c
char *argv[3];

argv[0] = "echo";
argv[1] = "hello";
argv[2] = 0;
exec("/bin/echo", argv);
printf("exec error\n");
```
- replaces calling program with instance of program `/bin/echo`
- has arg list `echo hello`
- usually, first element of arg array is just name of program -- ignored
- use above calls to run programs on behalf of users
- main structure of shell simple
	- read line of input with `getcmd`
	- call `fork` to create copy of process
	- parent calls `wait` while child runs
- why are `fork` and `exec` not in single call?
	- separation exploited in implementation of I/O redirection
	- wasteful to create duplicate process and immediately replace it
	- operating kernels can optimize this, by using virtual mem. techniques like copy-on-write
	- user-space memory allocated implicitly
		- `fork` allocates memory required for child's copy of parent
		- `exec` allocates enough memory to hold executable file
	- can get more memory with `sbrk(n)`
# i/o and file descriptors
- *file descriptor* is small integer representing kernel-managed object, that a process can read from or write to
- get this by opening file, directory, or device
	- or by making a pipe
	- or by duplicating existing one
- the object is just referred to as a "file"
	- interface abstracts away diff, they're all streams of bytes
	- refer to this as I/O
- file descriptor is an index into a per-process table
	- so each process has private space of file descriptors starting at 0
- convention:
	- read from f.d. 0 (standard input)
	- output written to f.d. 1 (standard output)
	- err written to f.d. 2 (standard error)
- this convention used to implement i/o redirection and pipelines
- shell ensures it always has three f.d.'s open, which are default f.d.'s
- `read` and `write` syscalls read bytes from, write bytes to open files named by these f.d.'s
- `read(fd, buf, n)` reads at most `n` bytes from file desc. `fd`, copies them into `buf`, return # of bytes read
- each f.d. that refers to a file has offset associated with it
	- `read` will get data from that offset, then advance it by # of bytes read
	- subsequent `read` will get the bytes following the first one
	- no more bytes? `read` returns 0 for EOF
- `write(fd, buf, n)` writes `n` bytes from `buf` to file desc `fd`, returns # of bytes written
- < `n` bytes written only when error occurs
- like `read`, `write` writes data at current file offset, then advances it by # of bytes written
- each `write` picks up where previous one left off
example program fragment, which is basically `cat`:
```c
char buf[512];
int n;

for(;;) {
	n = read(0, buf, sizeof buf);
	if (n == 0) break;
	
	if (n < 0) {
		fprintf(2, "read error\n");
		exit(1);
	}
	
	if (write(1, buf, n) != n) {
		fprintf(2, "write error\n");
		exit(1);
	}
}
```
- cat doesn't know what it's reading - it might be file, console, pipe
	- same deal with writing
	- we just have 0 for input and 1 for output
- `close` syscall releases a file descriptor, making it free for reuse for future `open`, `pipe`, or `dup` syscall
- newly allocated f.d. is always lowest-numbered unused descriptor (for current process)
- f.d.'s and `fork` interact to make i/o redirection easy
- `fork` copies parent's f.d. table along with memory, so child starts with same open files as parent
- syscall `exec` replaces calling process's mem. but preserves file table
- can get i/o redirection by forking, reopening chosen f.d.'s in the child, and then calling `exec` to run new program
example (`cat < input.txt`):
```c
char *argv[2];

argv[0] = "cat";
argv[1] = 0;
if (fork() == 0) {
	close(0);
	open("input.txt", O_RDONLY);
	exec("cat", argv);
}
```
- child (`fork()` returns 0) closes f.d. 0, so `open` is guaranteed to use that f.d. for newly opened `input.txt` (0 is smallest available)
	- `cat` executes with f.d. 0 (standard input) referring to input.txt
	- parent process's f.d.'s are not changed by this sequence, since only child's descriptors are modified
- second arg. to `open` is set of flags, expressed as bits, that control what `open` does
	- reading, writing, both, creation, or truncation
- why `fork` and `exec` separate calls?
	- shell can redirect child I/O without disturbing I/O of main shell
- underlying file offsets are shared
- `dup` syscall duplicates existing file descriptor
	- returns a new one, but referring to the *same* underlying object
	- they share an offset as well
- offset shared if they were derived from the same original file descriptor
	- otherwise, offsets are not shared
	- not even if `open` was called on the same file
	- so we can get commands like this
	- `ls existing-file non-existing-file > tmp1 2>&1`
		- `2>&1` tells shell to give the command a file descriptor 2 that is a duplicate of descriptor 1
		- both the name of the existing file, and the error message for the nonexisting one, will show up in tmp1
# pipes
- small kernel buffer exposed to processes as a pair of file descriptors
- one for reading, one for writing
- write to one end? can read from other end
- provide way for communication between processes
example (`wc`):
```c
int p[2];
char *argv[2];

argv[0] = "wc";
argv[1] = 0;
pipe(p);
if (fork() == 0) {
	close(0);
	dup(p[0]);
	close(p[0]);
	close(p[1]);
	exec("/bin/wc", argv);
} else {
	close(p[0]);
	write(p[1], "hello world\n", 12);
	close(p[1]);
}
```
- child
	- calls `close` and `dup` to have stdin (f.d. 0) refer to read end of pipe
	- file descriptors in `p` closed
	- calls `exec` to run `wc`
	- so then `wc` will read from the pipe for stdin
- parent
	- closes read side of pipe
	- writes to pipe
	- then closes write side
- no data? `read` on pipe will wait for data to be written, or for all file descriptors referring to write end to be closed
	- latter case: `read` returns 0, just like EOF
- `read` blocks until impossible for new data to arrive
	- child must close write end of pipe before executing `wc`
	- because of one of `wc`'s f.d.'s referred to write end, `wc` never sees EOF
- pipelines like `grep fork sh.c | wc -l`:
	- create pipe to connect left and right
	- right end can also include a pipe
	- so we can get a tree of processes
- why not just use temp files?
	- automatic cleanup -- without file redirection, have to remove `/tmp`
	- pipes can pass arbitrarily long streams of data
		- file redirection requires enough free space on disk to store all data
	- parallel execution of pipeline stages
		- file approach requires first program to finish before second starts
# file system
- `xv6` FS provides:
	- data files, uninterpreted byte arrays
	- directories, which have named refs to data files and other directories
- dirs form a tree, starting at the root
- a path like `a/b/c` refers to file/directory named `c` in `b` in `a` in root `/`
- if not begin with `/`, relative path -- evaluated relative to current directory of current process
- can change this with `chdir`
- more syscalls to create new files/directories
	- `mkdir` to create dir
	- `open` with `O_CREATE` creates new file
	- `mknod` creates new device file
- device file is special
	- has major and minor device numbers
		- passed as args to `mknod`
	- these uniquely identify kernel device
	- when opened, kernel diverts `read` and `write` syscalls to kernel device impl, instead of filesystem
- file name != file
	- same underlying file, called *inode*, can have multiple names
	- these are *links*
	- each link consists of entry in directory
	- entry has file name, ref to inode
- inode holds *metadata*, including:
	- type (file, directory, device)
	- length
	- location of file content on disk
	- number of links to file
- `fstat` syscall retrieves info from inode that the f.d. refers to
- `link` syscall creates another FS name, referring to same inode as existing file
	- aliases for the same thing
- each inode identified by unique inode number
- how to determine if same file?
	- call `fstat`, look at inode number `ino`
	- `nlink` count set to 2
- `unlink` syscall removes name from FS
	- but the inode and disk space holding content only free when link count 0, no f.d.'s refer to it
example (idiomatic way to create temp inode that's cleaned up when `fd` closed or process exits)
```c
fd = open("/tmp/xyz", O_CREATE|O_RDWR);
unlink("/tmp/xyz");
```
- file utilities are callable from the shell as user-level programs
- `mkdir`, `ln`, `rm`, etc
- can extend CLI by adding new user-level programs!
- `cd` is built into shell
	- must change working directory of shell itself
	- if run as regular command, then shell would:
		- fork child
		- child process runs `cd`, and the child changes
		- but the shell does not change!
# real world
- combo of "standard" FDs, pipes, convenient shell syntax
	- major advance in writing general-purpose reusable programs
- culture of "software tools"
- shell was first scripting language
- unix syscall interface standardized through POIS standard
- `xv6` not POSIX compliant, since missing many syscalls
- any OS needs to
	- multiplex processes onto underlying hardware
	- isolate processes
	- provide mechanisms for communication
	- 