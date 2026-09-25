#pragma once
// Error checking helpers shared by all examples.
//
// CUDA's runtime API reports failures through return codes, not exceptions, and
// almost every real bug shows up as a cudaError_t you forgot to inspect. Wrap
// every runtime call from the start -- it is the single habit that saves the
// most debugging time when learning.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n  in: %s\n",          \
                         __FILE__, __LINE__, cudaGetErrorString(err_),        \
                         #call);                                              \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// Kernel launches do not return an error code. Call this right after a launch:
// cudaGetLastError() catches bad launch configurations (too many threads, too
// much shared memory), and the synchronize catches faults raised while the
// kernel was actually executing.
#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        CUDA_CHECK(cudaGetLastError());                                       \
        CUDA_CHECK(cudaDeviceSynchronize());                                  \
    } while (0)

inline void print_device_info() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::printf("Device %d: %s\n", device, prop.name);
    std::printf("  compute capability : %d.%d  (compile with -arch=sm_%d%d)\n",
                prop.major, prop.minor, prop.major, prop.minor);
    std::printf("  SMs                : %d\n", prop.multiProcessorCount);
    std::printf("  global memory      : %.2f GiB\n",
                static_cast<double>(prop.totalGlobalMem) / (1024.0 * 1024.0 * 1024.0));
    std::printf("  shared mem / block : %zu KiB\n", prop.sharedMemPerBlock / 1024);
    std::printf("  max threads / block: %d\n", prop.maxThreadsPerBlock);
    std::printf("  warp size          : %d\n", prop.warpSize);
    std::printf("\n");
}
