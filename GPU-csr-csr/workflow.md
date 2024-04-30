# Workflow for the GPU sparse matrix matrix multiplication

- The inputs are two sparse matrices A and B
- Each thread should represent an entry (i,j) in the final matrix A @ B

- We should have a global array where each of the core will write to the array, storing the value they get from the calculation 
- Just remember that if a device function needs to access it, then we need to cast the pointer first

- If tid is longer than A.rows and B.cols, then we return because that thread is not going to do anything
- Otherwise, the thread can perform calculations 

- Now onto the spgemm_kernel, again a thread here computes a single entry, so the GPU is going to do a whole bunch of paralel reads, which is okay since they're writing to a bunch of different things
- I believe the write here also does not need to be synchronized because no two threads are writing to the same place