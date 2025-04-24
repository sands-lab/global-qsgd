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
    dev_gradient[i] = dev_gradient[i]/(*dev_global_norm);
    // Decode
    int is_zero = (dev_gradient[i] != 0);
    int exp;
    float prob = abs(frexpf(dev_gradient[i], &exp)) / 0.5 - 1.; // exp = [-127, 0]; prob = [0.5, 1) -> [0, 1)
    int round_down = (dev_rand[i] >= prob);
    exp = exp - round_down;
    exp = max(exp, -127);
    assert(exp <=-1 && exp >= -127); // exp = [-127, -1]
    exp = -exp; // exp = [1, 127] for positive and [129, 255] for negative
    int is_negative = (dev_gradient[i] < 0);
    exp += is_negative * 128; // Negative Highest bit = 1
    dev_compressed[i] = static_cast<uint8_t>(exp);
    dev_compressed[i] = dev_compressed[i] * is_zero; // Set to 0 if gradient is 0
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
    // Decode
    int exp = dev_compressed[i];
    int is_negative = (exp >= 128);
    exp = exp - is_negative * 128;
    dev_gradient[i] = pow(2,-exp);
    dev_gradient[i] = dev_gradient[i] * (1 - is_negative * 2);
    // DeNormalize
    dev_gradient[i] = dev_gradient[i] * (*dev_global_norm)/world_size;
    int is_zero = (exp == 0);
    dev_gradient[i] = dev_gradient[i] * (1 - is_zero);
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
    int is_negative_1 = (dev_compressed_a[i] >= 128);
    int is_negative_2 = (dev_compressed_b[i] >= 128);
    e_1 = dev_compressed_a[i] - is_negative_1 * 128;
    e_2 = dev_compressed_b[i] - is_negative_2 * 128;
    sign_1 = 1 - is_negative_1 * 2;
    sign_2 = 1 - is_negative_2 * 2;
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
    dev_compressed_a[i] += 128 * (sign_res == -1);
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

// 4-bit standard dithering compression
__global__ void standard_dithering_4bit_compress_kernel(
    float* input,
    uint8_t* output,
    float* global_min,
    float* global_max,
    const float* rand_vals,
    long long numel) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Scale to [0, 1] range
        float min_val = *global_min;
        float max_val = *global_max;
        float range = max_val - min_val;
        
        // Handle edge case where all values are the same
        if (range < 1e-6) {
            output[idx] = 0;
            return;
        }
        
        // Normalize to [0, 1]
        float normalized = (input[idx] - min_val) / range;
        
        // Scale to [0, 15] for 4-bit quantization
        float scaled = normalized * 15.0f;
        
        // Stochastic rounding
        float floor_val = floorf(scaled);
        float frac = scaled - floor_val;
        int quantized;
        
        if (rand_vals[idx] < frac) {
            quantized = (int)floor_val + 1;
        } else {
            quantized = (int)floor_val;
        }
        
        // Clamp to valid range [0, 15]
        quantized = min(15, max(0, quantized));
        
        // Store as uint8_t (will be packed later)
        output[idx] = static_cast<uint8_t>(quantized);
    }
}

// Pack two 4-bit values into a single byte
__global__ void pack_4bit_kernel(
    uint8_t* input,
    uint8_t* output,
    long long numel) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long out_idx = idx / 2;
    
    if (idx < numel && idx % 2 == 0) {
        uint8_t val1 = input[idx];
        uint8_t val2 = (idx + 1 < numel) ? input[idx + 1] : 0;
        
        // Pack: high 4 bits = first value, low 4 bits = second value
        output[out_idx] = (val1 << 4) | (val2 & 0x0F);
    }
}

// Unpack from a single byte to two 4-bit values
__global__ void unpack_4bit_kernel(
    uint8_t* input,
    uint8_t* output,
    long long numel) {
    
    const int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long in_idx = out_idx / 2;
    
    if (out_idx < numel) {
        uint8_t packed = input[in_idx];
        
        if (out_idx % 2 == 0) {
            // Extract high 4 bits
            output[out_idx] = (packed >> 4) & 0x0F;
        } else {
            // Extract low 4 bits
            output[out_idx] = packed & 0x0F;
        }
    }
}

// Tree reduction for 4-bit values with overflow detection
__global__ void reduce_4bit_kernel(
    uint8_t* a,
    uint8_t* b,
    uint8_t* overflow,
    long long numel) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Get the two values to reduce
        int val_a = static_cast<int>(a[idx]);
        int val_b = static_cast<int>(b[idx]);
        
        // Sum them
        int sum = val_a + val_b;
        
        // Check for overflow (beyond 4 bits)
        bool has_overflow = (sum > 15);
        
        // Record overflow
        overflow[idx] = has_overflow ? 1 : 0;
        
        // Handle overflow: clamp to 15
        sum = min(15, sum);
        
        // Store back
        a[idx] = static_cast<uint8_t>(sum);
        b[idx] = static_cast<uint8_t>(sum);
    }
}

// 4-bit standard dithering decompression
__global__ void standard_dithering_4bit_decompress_kernel(
    uint8_t* input,
    float* output,
    float* global_min,
    float* global_max,
    long long numel) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Get quantized value
        int quantized = static_cast<int>(input[idx]);
        
        // Scale back from [0, 15] to [0, 1]
        float normalized = quantized / 15.0f;
        
        // Convert back to original range
        float min_val = *global_min;
        float max_val = *global_max;
        float range = max_val - min_val;
        
        output[idx] = normalized * range + min_val;
    }
}

// Main interface for 4-bit compression
torch::Tensor standard_dithering_4bit_compress_cuda(
    torch::Tensor input,
    torch::Tensor global_min,
    torch::Tensor global_max) {
    
    // Get device and tensor info
    cudaSetDevice(input.device().index());
    long long numel = input.numel();
    
    // Allocate temporary storage for unpacked 4-bit values
    auto temp_storage = torch::empty(numel, torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
    
    // Generate random values for stochastic rounding
    auto rand_vals = torch::rand_like(input);
    
    // Calculate grid and block dimensions
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    // Compress values to 4-bit (stored in 8-bit)
    standard_dithering_4bit_compress_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(),
        temp_storage.data_ptr<uint8_t>(),
        global_min.data_ptr<float>(),
        global_max.data_ptr<float>(),
        rand_vals.data_ptr<float>(),
        numel
    );
    
    // Determine size of packed output (ceil(numel/2))
    long long packed_size = (numel + 1) / 2;
    auto output = torch::empty(packed_size, torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
    
    // Calculate new grid and block dimensions for packing
    threads = 256;
    blocks = (numel + threads - 1) / threads;
    
    // Pack two 4-bit values into each byte
    pack_4bit_kernel<<<blocks, threads>>>(
        temp_storage.data_ptr<uint8_t>(),
        output.data_ptr<uint8_t>(),
        numel
    );
    
    return output;
}

// Main interface for tree reduction
std::tuple<torch::Tensor, torch::Tensor> standard_dithering_4bit_reduce_cuda(
    torch::Tensor a,
    torch::Tensor b,
    float original_numel) {
    
    // Cast original_numel to long long for internal use
    long long numel_ll = static_cast<long long>(original_numel);
    
    auto numel = a.numel();
    auto options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto overflow = torch::zeros({numel}, options);
    
    // Allocate temporary storage for unpacked 4-bit values
    auto a_unpacked = torch::empty(numel, options);
    auto b_unpacked = torch::empty(numel, options);
    
    // Calculate grid and block dimensions
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    // Unpack both tensors
    unpack_4bit_kernel<<<blocks, threads>>>(
        a.data_ptr<uint8_t>(),
        a_unpacked.data_ptr<uint8_t>(),
        numel
    );
    
    unpack_4bit_kernel<<<blocks, threads>>>(
        b.data_ptr<uint8_t>(),
        b_unpacked.data_ptr<uint8_t>(),
        numel
    );
    
    // Perform reduction
    reduce_4bit_kernel<<<blocks, threads>>>(
        a_unpacked.data_ptr<uint8_t>(),
        b_unpacked.data_ptr<uint8_t>(),
        overflow.data_ptr<uint8_t>(),
        numel
    );
    
    // Return the reduced tensor (a_unpacked) and overflow flag
    return std::make_tuple(a_unpacked, overflow);
}

// Main interface for 4-bit decompression
torch::Tensor standard_dithering_4bit_decompress_cuda(
    torch::Tensor input,
    torch::Tensor global_min,
    torch::Tensor global_max,
    float original_numel) {
    
    // Cast original_numel to long long for internal use
    long long numel_ll = static_cast<long long>(original_numel);
    
    // Each byte contains 2 values
    auto numel = input.numel() * 2;
    
    // Create output tensor with size of original tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(input.device());
    auto output = torch::zeros({numel_ll}, options);
    
    // Allocate temporary storage for unpacked 4-bit values
    auto unpacked = torch::empty(numel, torch::TensorOptions().device(input.device()).dtype(torch::kUInt8));
    
    // Calculate grid and block dimensions
    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    
    // Unpack the input
    unpack_4bit_kernel<<<blocks, threads>>>(
        input.data_ptr<uint8_t>(),
        unpacked.data_ptr<uint8_t>(),
        numel
    );
    
    // Decompress to original range
    standard_dithering_4bit_decompress_kernel<<<blocks, threads>>>(
        unpacked.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        global_min.data_ptr<float>(),
        global_max.data_ptr<float>(),
        numel_ll
    );
    
    return output;
}
