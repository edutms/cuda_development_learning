// Example 01: the smallest possible kernel launch.
//
// Proves the whole toolchain works: nvcc compiled it, the driver accepted it,
// and threads on the GPU actually ran. Device-side printf output is buffered
// and only flushed when the device synchronizes, which is why the launch is
// followed by CUDA_CHECK_KERNEL().

#include <cstdio>
#include "cuda_check.cuh"

__global__ void hello_kernel() {
    // Every thread computes its own global index from its position in the grid.
    // This indexing pattern is the foundation of essentially every CUDA kernel.
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    printf("Hello from block %d, thread %d (global id %d)\n",
           blockIdx.x, threadIdx.x, global_id);
}

int main() {
    print_device_info();

    const int blocks = 2;
    const int threads_per_block = 4;

    std::printf("Launching %d blocks x %d threads = %d threads\n\n",
                blocks, threads_per_block, blocks * threads_per_block);

    hello_kernel<<<blocks, threads_per_block>>>();
    CUDA_CHECK_KERNEL();

    std::printf("\nDone.\n");
    return 0;
}
