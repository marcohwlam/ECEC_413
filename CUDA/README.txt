Report the speedup achieved by the GPU kernels over the CPU implementation for the following
matrix sizes: 512 × 512, 1024 × 1024, and 2048 × 2048.

speedup = GPU time / CPU time
Speed up 512 x 512
global memory:
shared memory: 1489

Speed up 1024 x 1024
global memory:
shared memory: 424

Speed up 2048 x 2048
global memory:
shared memory: 132.3


• Include a brief report describing how you designed your kernels, using code or pseudocode
to clarify the discussion, and the speedup obtained over the serial version for both GPUbased
versions.



• The GTX 1080 GPU can achieve a peak processing rate of about 8800 GFLOPs. The memory
bandwidth on the device is 320 GB/s. How many floating-point operations must be performed
per load operation to achieve the peak processing rate? What is the performance of
your kernels (both naive as well as the one that uses shared memory), in terms of GFLOPs

Too optimize the processing rate the ratio of arithmetic operation / memory IO operation
must close to peak processing rate / memory bandwidth

We convert speed of loading floating point to memory
each floating point need 32bit which is 4 byte
320GB/s /4 = 80 GFLOPs

ratio of arithmetic operation/ memory IO operation in the kernels should be
8800/80 = 110
to optimize the processing, each mem io operation should have 110 arithmetic operation
