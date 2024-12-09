#include <torch/torch.h>
#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor> find_product_sparsity(torch::Tensor _input_act, int tile_size_m, int tile_size_k);

void find_product_sparsity_cpp(uint8_t* _input_act, uint8_t* prosparsity_act, int* prefix_array, int m, int k, int tile_size_m, int tile_size_k);