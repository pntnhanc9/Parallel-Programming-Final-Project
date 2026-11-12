// cuda_utils.cuh -- CUDA error checking and GPU timing (for V1, V2, ...).
// NOT included by V0 / main.cu yet, so the sequential baseline stays CPU-only.
// Authors: <TODO: names / student IDs>
// External sources: CHECK macro pattern from the course CUDA lab material.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t error_ = (call);                                         \
    if (error_ != cudaSuccess) {                                               \
      fprintf(stderr, "Error: %s:%d, code: %d, reason: %s\n", __FILE__,        \
              __LINE__, (int)error_, cudaGetErrorString(error_));              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// CUDA-event timer. Usage: t.Start(); <launch work>; t.Stop(); ms = t.Elapsed();
// Elapsed() synchronizes on the stop event, so it is safe with async launches.
struct GpuTimer {
  cudaEvent_t start, stop;

  GpuTimer() {
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
  }
  ~GpuTimer() {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }
  GpuTimer(const GpuTimer &) = delete;
  GpuTimer &operator=(const GpuTimer &) = delete;

  void Start() { CHECK(cudaEventRecord(start, 0)); }
  void Stop() { CHECK(cudaEventRecord(stop, 0)); }
  float Elapsed() {
    float ms = 0.f;
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
  }
};
