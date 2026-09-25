// Example 02: the full host/device cycle.
//
//   allocate on host -> allocate on device -> copy H2D -> launch -> copy D2H
//   -> verify on CPU -> free everything
//
// Every CUDA program you write is a variation on this shape. It also times the
// kernel with CUDA events, which is the correct way to measure GPU work (host
// clocks would measure the asynchronous launch, not the execution).

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_check.cuh"

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // The grid is sized *up* to cover n, so the last block usually contains
    // threads with i >= n. Without this guard they would write out of bounds.
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    print_device_info();

    const int n = 1 << 20;                    // 1,048,576 elements
    const size_t bytes = n * sizeof(float);

    // ---- host data ----
    std::vector<float> h_a(n), h_b(n), h_c(n);
    for (int i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // ---- device data ----
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // ---- launch configuration ----
    // The (n + tpb - 1) / tpb idiom is integer ceiling division: round the
    // block count up so that every element gets a thread.
    const int threads_per_block = 256;
    const int blocks = (n + threads_per_block - 1) / threads_per_block;

    std::printf("n = %d, %d blocks x %d threads\n\n", n, blocks, threads_per_block);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK_KERNEL();
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    // ---- verify on the CPU ----
    int errors = 0;
    for (int i = 0; i < n; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            if (errors < 5) {
                std::printf("  mismatch at %d: got %f, expected %f\n",
                            i, h_c[i], expected);
            }
            ++errors;
        }
    }

    std::printf("kernel time    : %.3f ms\n", ms);
    // 3 accesses per element (read a, read b, write c), 4 bytes each.
    std::printf("effective bw   : %.1f GB/s\n",
                (3.0 * bytes / 1e9) / (ms / 1e3));
    std::printf("result         : %s", errors == 0 ? "PASSED\n" : "FAILED\n");
    if (errors) std::printf("  %d mismatches\n", errors);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return errors == 0 ? 0 : 1;
}
