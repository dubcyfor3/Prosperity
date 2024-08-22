#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "prosparsity_cuda.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("find_product_sparsity", &find_product_sparsity, "get matrix with product sparsity");
}