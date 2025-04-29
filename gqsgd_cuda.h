#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <random>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <bitset>

#define CUDA_CHECK(condition)                                                  \
do {                                                                           \
  cudaError_t cuda_result = condition;                                         \
  if (cuda_result != cudaSuccess) {                                            \
    printf("%s on line %i in %s returned: %s(code:%i)\n", #condition,          \
           __LINE__, __FILE__, cudaGetErrorString(cuda_result),                \
           cuda_result);                                                       \
    throw std::runtime_error(                                                  \
        std::string(#condition) + " in file " + __FILE__                       \
        + " on line " + std::to_string(__LINE__) +                             \
        " returned: " + cudaGetErrorString(cuda_result));                      \
  }                                                                            \
} while (0)

__inline__ float *dev_prob_generator(){
    extern const int num_elem;
    float *dev_rand;
    float *host_rand = new float[num_elem];
    std::srand(std::time(nullptr));
    for (int i = 0; i < num_elem; i++) {
        host_rand[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    }
    cudaMalloc(&dev_rand, sizeof(float)*num_elem);
    cudaMemcpy(dev_rand, host_rand, sizeof(float)*num_elem, cudaMemcpyHostToDevice);
    return dev_rand;
}