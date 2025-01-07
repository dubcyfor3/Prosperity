import pickle
import torch
import os
import torch
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from utils_ana import get_density, ceil_a_by_b, img2col
from collections import defaultdict
import torch.nn.functional as F
import time
def calculate_sparsity(tensor):
    """Calculate the sparsity of a given tensor."""
    tensor_np = tensor.cpu().numpy()  # Convert tensor to NumPy array
    num_nonzero_elements = (tensor_np != 0).sum()  # Count non-zero elements
    total_elements = tensor_np.size  # Total number of elements in the tensor
    sparsity = (num_nonzero_elements / total_elements)
    return sparsity

def product_sparsify_whole_matrix(input_tensor: torch.Tensor, tile_size_m = 256, tile_size_k = 16):
    input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
    tile_size_k = tile_size_k
    tile_size_m = tile_size_m
    sparsified_tensor = torch.zeros_like(input_tensor)
    num_prefix = 0

    print("sparsity: ", get_density(input_tensor))

    for m_id in range(0, input_tensor.shape[0], tile_size_m):
        for k_id in range(0, input_tensor.shape[1], tile_size_k):
            
            cur_tile_size_m = min(tile_size_m, input_tensor.shape[0] - m_id)
            cur_tile_siz_k = min(tile_size_k, input_tensor.shape[1] - k_id)
            
            tile = input_tensor[m_id:m_id+cur_tile_size_m, k_id:k_id+cur_tile_siz_k]

            sparsified_tile = tile.clone()
            for i in range(tile.shape[0]):
                cur_row = tile[i]
                nnz = torch.sum(cur_row != 0).item()
                if nnz < 2:
                    continue

                and_result = torch.logical_and(cur_row, tile)
                equalities = torch.eq(and_result, tile)
                is_subset = torch.all(equalities, dim=-1)

                equalities = torch.eq(cur_row, tile)
                is_equal = torch.all(equalities, dim=-1)
                is_bigger_index = torch.arange(tile.shape[0]) >= i

                is_excluded = torch.logical_and(is_equal, is_bigger_index)
                is_real_subset = torch.logical_and(is_subset, ~is_excluded)

                if torch.sum(is_real_subset) == 0:
                    continue


                subset_row = tile[is_real_subset]
                subset_row_nnz = torch.sum(subset_row != 0, dim=-1)
                max_subset_size = torch.max(subset_row_nnz).item()
                max_subset = subset_row[torch.argmax(subset_row_nnz)]
                if max_subset_size > 0: # can also reuse even when the size is 1
                    sparsified_tile[i] = torch.logical_xor(sparsified_tile[i], max_subset)
                    num_prefix += 1

            sparsified_tensor[m_id:m_id+cur_tile_size_m, k_id:k_id+cur_tile_siz_k] = sparsified_tile

    # print("sparsity: ", get_density(sparsified_tensor))
    return sparsified_tensor, num_prefix

def process_tensor(tensor):
    # 确保输入是四维张量
    assert tensor.dim() == 4, "Input tensor must be 4-dimensional"
    
    # 沿着第0个维度（batch维度）求和
    summed = torch.sum(tensor, dim=0)
    
    # 将结果与1比较，大于1的设为1，小于等于1的设为0
    binarized = (summed > 1).float()
    
    # 增加一个维度，使结果形状为 [1, C, H, W]
    result = binarized.unsqueeze(0)
    
    return result

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_name = 'vgg16_cifar10_loas.pkl'
    model = 'vgg16'
    with open(file_name, 'rb') as f:
        data1 = pickle.load(f)
    
    if model == 'resnet19':
        conv_params = [
            # skip the first layer
            {"stride": (2, 2), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (2, 2), "padding": (0, 0), "kernel_size": (1, 1)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (2, 2), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (2, 2), "padding": (0, 0), "kernel_size": (1, 1)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (2, 2), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (2, 2), "padding": (0, 0), "kernel_size": (1, 1)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
        ]
    else:
        conv_params = [
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
            {"stride": (1, 1), "padding": (1, 1), "kernel_size": (3, 3)},
        ]

    # Check and calculate sparsity for Conv2d layers
    if 'Conv2d' in data1:
        conv_layers = data1['Conv2d']
        print(f"Found {len(conv_layers)} Conv2d layers.")

        total_elements = 0
        total_nnzs = 0
        # Iterate over each Conv2d layer and calculate sparsity
        for i, tensor in enumerate(conv_layers):
            if i == 0:
                continue
            tensor = process_tensor(tensor)
            H, W = tensor.shape[-2:]
            sparsity = calculate_sparsity(tensor)
            print(f"Sparsity of Conv2d layer {i }: {(1-sparsity) * 100:.2f}%")
            params = conv_params[i-1]
            tensor = F.unfold(tensor, kernel_size=params['kernel_size'], padding=params['padding'], stride=params['stride'])
            tensor.squeeze(0)

            prosperity_tensor, num_prefix = product_sparsify_whole_matrix(tensor, tile_size_k=16)
            prosperity_tensor.unsqueeze(0)
            F.fold(prosperity_tensor, (H,W) , kernel_size=params['kernel_size'], padding=params['padding'], stride=params['stride'])
            sparsity = calculate_sparsity(prosperity_tensor)
            print(f"Sparsity of Conv2d layer Prosperity {i }: {(1-sparsity) * 100:.2f}%")

            total_elements += torch.numel(prosperity_tensor)
            total_nnzs += torch.sum(prosperity_tensor != 0).item()
        print(f"Total sparsity: {(1 - total_nnzs / total_elements) * 100:.2f}%")