#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include "gqsgd_cuda.h"
const long long MAX_THREAD_PER_BLOCK = 1024;
const long long MAX_NUMBER_OF_BLOCK = 65535;
int threads;
int blocks;
// Standdard Dithering Random Rounding
__global__ void standard_dithering_random_round_cuda_kernel(
  float* __restrict__ input,
  const float* __restrict__ rand,
  long long len) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < len) {
       input[index] = ( rand[index] < (input[index] - floor(input[index]) ) ) ? ceil(input[index]) : floor(input[index]);
  }
}

torch::Tensor standard_dithering_random_round_cuda(torch::Tensor input) {
  auto rand = torch::rand_like(input, torch::TensorOptions().device(input.device())); // [0, 1)
  const int threads = 1024;
  long long numel = input.numel();
  auto blocks = numel/threads;
  if (numel%threads || !blocks) blocks++;
  standard_dithering_random_round_cuda_kernel<<<blocks, threads>>>(
    input.data_ptr<float>(),
    rand.data_ptr<float>(),
    numel);

  return input;
}

// Exponential Dithering
#define max_interval 127.0 + 1

__global__ void exponential_dithering__compress_cuda_kernel(
    float *dev_gradient,
    uint8_t *dev_compressed,
    const float*  dev_rand,
    float *dev_global_norm,
    long long dev_num_elem) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < dev_num_elem; i += stride) {
     // Normalize -> [-1,1]
    // printf("dev_gradient[i] is %f, dev_global_norm is %f", dev_gradient[i], *dev_global_norm);
    dev_gradient[i] = dev_gradient[i]/(*dev_global_norm);
    // Decode
    if (dev_gradient[i] == 0) {
      dev_compressed[i] = 0;
    }else{
      int exp;
      float prob = abs(frexpf(dev_gradient[i], &exp)) / 0.5 - 1.; // exp = [-127, 0]; prob = [0.5, 1) -> [0, 1)
      if (dev_rand[i] >= prob) exp = exp - 1.0;// Prob < 1 so only round to 2^exp or 2^exp-1
      exp = max(exp, -127);
      assert(exp <=-1 && exp >= -127); // exp = [-127, -1]
      exp = -exp; // exp = [1, 127] for positive and [129, 255] for negative
      if (dev_gradient[i] < 0) exp += 128; // Negative Highest bit = 1
      dev_compressed[i] = static_cast<uint8_t>(exp);
    }
  }
}

__global__ void exponential_dithering__decompress_cuda_kernel(
    uint8_t *dev_compressed,
    float *dev_gradient,
    float *dev_global_norm,
    long long dev_num_elem,
    int world_size) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < dev_num_elem; i += stride) {
    // Decode
    int exp = dev_compressed[i];
    if(exp>=128){
      exp = exp - 128;
    }
    if(exp==0) {
      dev_gradient[i] = 0.0;
      return;
    }
    dev_gradient[i] = pow(2,-exp);
    if(dev_compressed[i]>=128) dev_gradient[i] = -dev_gradient[i];
    // DeNormalize
    dev_gradient[i] = dev_gradient[i] * (*dev_global_norm)/world_size;
  }
}

__global__ void exponential_dithering__reduce_cuda_kernel(
    uint8_t *dev_compressed_a,
    uint8_t *dev_compressed_b,
    float *dev_rand,
    long long dev_num_elem) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int stride = gridDim.x * blockDim.x;
  for (unsigned int i = tid; i < dev_num_elem; i += stride) {
    // Decode
    int sign_1, sign_2, e_1, e_2;
    {
      if (dev_compressed_a[i] >= 128 ){
        sign_1 = -1;
        e_1 = dev_compressed_a[i] - 128;
      } else{
        sign_1 = 1;
        e_1 = dev_compressed_a[i];
      }
      if (dev_compressed_b[i] >= 128 ){
        sign_2 = -1;
        e_2 = dev_compressed_b[i] - 128;
      } else{
        sign_2 = 1;
        e_2 = dev_compressed_b[i];
      }
    }
    // Reduce
    int k = -floor(log2( 
        pow(2,-max_interval) +
        max(0.0 , dev_rand[i]-pow(2,-max_interval))
    ));
    int e_1_is_not_zero = e_1 > 0;
    int e_2_is_not_zero = e_2 > 0;
    int e_2_is_zero = 1 - e_2_is_not_zero;
    int sign_12 = sign_1 * sign_2 * e_1_is_not_zero * e_2_is_not_zero;
    // printf("sign_12 is %d, sign_1 is %d, sign_2 is %d, e_1 is %d, e_2 is %d, k is %d, rand is %f\n", sign_12, sign_1, sign_2, e_1, e_2, k, dev_rand[i]);
    // printf("int(e_1 <= e_2) = %d, e_2_is_zero = %d, e_1_is_not_zero = %d\n", int(e_1 <= e_2), e_2_is_zero, e_1_is_not_zero);
    int leq = (int(e_1 <= e_2) + e_2_is_zero) * e_1_is_not_zero;
    int sign_res = sign_1 * leq + sign_2 * (1 - leq);
    int non_zero = 1 - int(e_1==e_2) * int(sign_12==-1);
    int diff = abs(e_1 - e_2) - (1 - sign_12)/2;
    int e_res = (e_1 * leq + e_2 * (1 - leq) - sign_12 * int(k>diff)) * non_zero;
    // printf("e_1 is %d, e_2 is %d, sign_1 is %d, sign_2 is %d, sign_12 is %d, sign_res is %d, e_res is %d, k is %f, diff is %d, leq is %d, non_zero is %d\n", e_1, e_2, sign_1, sign_2, sign_12, sign_res, e_res, k, diff, leq, non_zero);
    //Encode
    dev_compressed_a[i] = e_res;
    if(sign_res == -1) dev_compressed_a[i] += 128;
    dev_compressed_b[i] = dev_compressed_a[i];
  }
}

// Quantize float32 gradient to uint_8 gradient
torch::Tensor exponential_dithering_compress_cuda(torch::Tensor input, torch::Tensor global_norm){
    cudaSetDevice(input.device().index());
    auto output = torch::empty_like(input, torch::TensorOptions().dtype(torch::kUInt8).device(input.device()));
    auto rand = torch::rand_like(input, torch::TensorOptions().device(input.device()));
    long long numel = input.numel();
    threads = min(MAX_THREAD_PER_BLOCK, numel);
    blocks = min( (numel + threads - 1)/threads, MAX_NUMBER_OF_BLOCK);
    assert(threads > 0); assert(blocks > 0);
    exponential_dithering__compress_cuda_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<uint8_t>(), rand.data_ptr<float>(), global_norm.data_ptr<float>(), numel);
    cudaDeviceSynchronize();
    return output;
}
// dequantize uint8 gradient to float32 gradient
torch::Tensor exponential_dithering_decompress_cuda(torch::Tensor input, torch::Tensor global_norm, int world_size){
    cudaSetDevice(input.device().index());
    auto output = torch::empty_like(input, torch::TensorOptions().dtype(torch::kFloat32).device(input.device()));
    long long numel = input.numel();
    threads = min(MAX_THREAD_PER_BLOCK, numel);
    blocks = min( (numel + threads - 1)/threads, MAX_NUMBER_OF_BLOCK);
    assert(threads > 0); assert(blocks > 0);
    exponential_dithering__decompress_cuda_kernel<<<blocks, threads>>>(input.data_ptr<uint8_t>(), output.data_ptr<float>(), global_norm.data_ptr<float>(), numel, world_size);
    cudaDeviceSynchronize();
    return output;
}

// // Aggregate two uint8 exponent gradient
torch::Tensor exponential_dithering_reduce_cuda(torch::Tensor input_a, torch::Tensor input_b){
    cudaSetDevice(input_a.device().index());
    auto rand = torch::rand_like(input_a, torch::TensorOptions().dtype(torch::kFloat32).device(input_a.device()));
    long long numel = input_a.numel();
    threads = min(MAX_THREAD_PER_BLOCK, numel);
    blocks = min( (numel + threads - 1)/threads, MAX_NUMBER_OF_BLOCK);
    assert(threads > 0); assert(blocks > 0);
    exponential_dithering__reduce_cuda_kernel<<<blocks, threads>>>(input_a.data_ptr<uint8_t>(), input_b.data_ptr<uint8_t>(), rand.data_ptr<float>(), numel);
    cudaDeviceSynchronize();
    return input_a;
}
