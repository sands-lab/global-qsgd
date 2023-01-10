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
#define max_interval 64.0 //63+1 largest_interval + ⌈log(n)⌉

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
    int exp;
    float prob = abs(frexpf(dev_gradient[i], &exp)) / 0.5 - 1.; // exp = [-127, 1]; prob = [0.5, 1) -> [0, 1)
    if (dev_rand[i] >= prob) exp = exp - 1.0;// Prob < 1 so only round to 2^exp or 2^exp-1
    exp = max(exp, -63);
    exp = min(exp, 0); //exp = [-63,0]
    exp = -exp; //exp = [0,63]
    if (dev_gradient[i] < 0) exp += 128; // Negative Highest bit = 1
    if (dev_gradient[i] == 0) exp = 127;  // Set exp of 0 to -63
    dev_compressed[i] = static_cast<uint8_t>(exp);
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
    if(dev_compressed[i]>=128) exp -= 128;
    dev_gradient[i] = pow(2,-exp);
    if(exp>=127) dev_gradient[i] = 0;
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
    int sign_a, sign_b, exp_a, exp_b;
    {
      if (dev_compressed_a[i] >= 128 ){
        sign_a = -1;
        exp_a = dev_compressed_a[i] - 128;
      } else{
        sign_a = 1;
        exp_a = dev_compressed_a[i];
      }
      if (dev_compressed_b[i] >= 128 ){
        sign_b = -1;
        exp_b = dev_compressed_b[i] - 128;
      } else{
        sign_b = 1;
        exp_b = dev_compressed_b[i];
      }
    }
    exp_a = - exp_a;
    exp_b = - exp_b;
    // Reduce
    int k = floor(log( 
            pow(2,-max_interval) +
            max(0.0 , dev_rand[i]-pow(2,-max_interval))
        ));
    int max_exp = max(exp_a, exp_b);
    int min_exp = min(exp_a, exp_b);
    int sign_ab = sign_a * sign_b;
    int diff = min_exp - max_exp + sign_ab - 1;
    int nonz = 1 - (exp_a == exp_b && sign_a != sign_b);
    int geq = (exp_a >= exp_b);
    int le = 1 - geq;
    int minz = (min_exp < 0);
    int reduce_sign = sign_a * geq + sign_b * le;
    int reduce_exp = nonz * (max_exp + minz * sign_ab * static_cast<int>(k<=diff));
    //Encode
    dev_compressed_a[i] = -reduce_exp;
    if(reduce_sign == -1) dev_compressed_a[i] += 128;
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
    return input_a;
}
