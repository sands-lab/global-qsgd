#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <tuple>

namespace py = pybind11;

const int num_elem = 100000;
int thread_per_block;
int num_block;
int* dev_num_elem;
const int MAX_THREAD_PER_BLOCK = 1024;
const int MAX_NUMBER_OF_BLOCK = 65535;

// CUDA forward declarations
torch::Tensor standard_dithering_random_round_cuda(torch::Tensor input);
torch::Tensor exponential_dithering_compress_cuda(torch::Tensor input, torch::Tensor global_norm);
torch::Tensor exponential_dithering_decompress_cuda(torch::Tensor input, torch::Tensor global_norm, int world_size);
torch::Tensor exponential_dithering_reduce_cuda(torch::Tensor input_a, torch::Tensor input_b);

// CUDA forward declarations for 4-bit standard dithering
torch::Tensor standard_dithering_4bit_compress_cuda(
    torch::Tensor input,
    torch::Tensor global_min,
    torch::Tensor global_max);

std::tuple<torch::Tensor, torch::Tensor> standard_dithering_4bit_reduce_cuda(
    torch::Tensor a,
    torch::Tensor b,
    long long original_numel);

torch::Tensor standard_dithering_4bit_decompress_cuda(
    torch::Tensor input,
    torch::Tensor global_min,
    torch::Tensor global_max,
    long long original_numel);

// C++ interface for exponential dithering
torch::Tensor standard_dithering_random_round(torch::Tensor input) {
  return standard_dithering_random_round_cuda(input);
}
torch::Tensor exponential_dithering_compress(torch::Tensor input, torch::Tensor global_norm) {
  return exponential_dithering_compress_cuda(input, global_norm);
}
torch::Tensor exponential_dithering_decompress(torch::Tensor input, torch::Tensor global_norm,int world_size) {
  return exponential_dithering_decompress_cuda(input, global_norm,world_size);
}
torch::Tensor exponential_dithering_reduce(torch::Tensor input_a, torch::Tensor input_b) {
  return exponential_dithering_reduce_cuda(input_a, input_b);;
}

// C++ interface for 4-bit standard dithering
torch::Tensor standard_dithering_4bit_compress(torch::Tensor input, torch::Tensor global_min, torch::Tensor global_max) {
  return standard_dithering_4bit_compress_cuda(input, global_min, global_max);
}
std::tuple<torch::Tensor, torch::Tensor> standard_dithering_4bit_reduce(torch::Tensor a, torch::Tensor b, long long original_numel) {
  return standard_dithering_4bit_reduce_cuda(a, b, original_numel);
}
torch::Tensor standard_dithering_4bit_decompress(torch::Tensor input, torch::Tensor global_min, torch::Tensor global_max, long long original_numel) {
  return standard_dithering_4bit_decompress_cuda(input, global_min, global_max, original_numel);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("standard_dithering_random_round", &standard_dithering_random_round, "standard dithering random round (CUDA)");
  m.def("exponential_dithering_compress", &exponential_dithering_compress, "exponential dithering compress");
  m.def("exponential_dithering_decompress", &exponential_dithering_decompress, "exponential dithering decompress");
  m.def("exponential_dithering_reduce", &exponential_dithering_reduce, "exponential dithering reduce");
  
  // Register 4-bit standard dithering functions
  m.def("standard_dithering_4bit_compress", &standard_dithering_4bit_compress, "4-bit standard dithering compress", 
        py::arg("input"), py::arg("global_min"), py::arg("global_max"));
  m.def("standard_dithering_4bit_reduce", &standard_dithering_4bit_reduce, "4-bit standard dithering reduce", 
        py::arg("a"), py::arg("b"), py::arg("original_numel"));
  m.def("standard_dithering_4bit_decompress", &standard_dithering_4bit_decompress, "4-bit standard dithering decompress", 
        py::arg("input"), py::arg("global_min"), py::arg("global_max"), py::arg("original_numel"));
}