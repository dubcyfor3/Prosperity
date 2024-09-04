from networks import FC, Conv2D, MaxPool2D, LIFNeuron, Attention, create_network, conv2d_2_fc
import torch
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from utils import get_density, ceil_a_by_b
from collections import defaultdict

import seaborn as sns
import heapq
import pickle
from sklearn.cluster import KMeans

def binary_weighted_k_meansplusplus(pattern_dict: dict, k: int = 256, tile_size: int = 32, max_iter: int = 10):
    # Filter out patterns with less than 2 nonzeros
    pattern_dict = {k: v for k, v in pattern_dict.items() if len(k) > 1}
    
    # Sort the pattern according to value
    pattern_dict = dict(sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True))

    # Convert patterns to vectors
    nodes = []
    for key in pattern_dict.keys():
        vec = torch.zeros(tile_size).to(torch.float32)
        for index in key:
            vec[index] = 1.0
        nodes.append(vec)  # Convert to numpy array

    if len(nodes) < k:
        centroids = torch.zeros(k, tile_size).to(torch.float32)
        for i in range(k):
            if i < len(nodes):
                centroids[i] = nodes[i].clone()
            else:
                centroids[i] = torch.zeros(tile_size).to(torch.float32)
        return centroids
    
    nodes = torch.stack(nodes)

    weights = list(pattern_dict.values())

    # Run K-Means with K-Means++ initialization
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=max_iter, n_init=1, random_state=42)
    kmeans.fit(nodes, sample_weight=weights)

    # Get the resulting centroids and round them
    centroids = kmeans.cluster_centers_
    centroids = np.round(centroids).astype(np.bool_)

    # Convert the centroids back to a PyTorch tensor
    centroids = torch.tensor(centroids)

    return centroids

def find_patterns_t(tensor: torch.Tensor, max_num_pattern, cur_tile_size_k, T):
    num_time_slices = tensor.shape[0] // T
    tensor_grouped = tensor.view(num_time_slices, T, -1)  # 重新排列形状
    tensor_mean = tensor_grouped.float().mean(dim=1)
    tensor_mean = tensor_mean > 0.5
    pattern_dict = defaultdict(int)
    for i in range(tensor.shape[0]):
        row = tensor[i]
        # convert row into nonzeros
        nonzeros = torch.nonzero(row).flatten()
        # convert to tuple
        nonzeros = tuple(nonzeros.tolist())
        pattern_dict[nonzeros] += 1
    
    pattern_tensor_kmeanspp = binary_weighted_k_meansplusplus(pattern_dict, k=max_num_pattern, tile_size=cur_tile_size_k)

    return pattern_tensor_kmeanspp

def pattern_analysis_test(tensor: torch.Tensor, pattern_tensor_kmeanspp):

    cycles_kmeanspp = 0
    optimal_cycles = 0
    for i in range(tensor.shape[0]):
        row = tensor[i]
        nnz = torch.sum(row != 0).item()
        # perform bitwise xor with the selected pattern
        # hamming_dist_kmeans = torch.sum(torch.logical_xor(pattern_tensor_kmeans, row), dim=-1)
        # smallest_dist_kmeans = torch.min(hamming_dist_kmeans).item()
        hamming_dist_kmeanspp = torch.sum(torch.logical_xor(pattern_tensor_kmeanspp, row), dim=-1)
        smallest_dist_kmeanspp = torch.min(hamming_dist_kmeanspp).item()
        # if smallest_dist_kmeans + 1 < nnz:
        #     cycles_kmeans += smallest_dist_kmeans + 1
        # else:
        #     cycles_kmeans += nnz
        if smallest_dist_kmeanspp + 1 < nnz:
            cycles_kmeanspp += smallest_dist_kmeanspp + 1
        else:
            cycles_kmeanspp += nnz
        if nnz > 0:
            optimal_cycles += 1

    return cycles_kmeanspp, optimal_cycles

def product_sparsify_whole_matrix_cuda(input_tensor: torch.Tensor, tile_size_m = 256, tile_size_k = 16):
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

if __name__ == '__main__':
    
    # model = 'spikformer'
    model = 'spikformer'
    dataset = 'cifar10'
    
    batch_size = 128
    T = 4
    tile_size_k = 16
    max_num_pattern = 512
    generate_pattern = True  # 设置为True以重新生成patterns

    nn_test = create_network(model, 'data/{}_{}_test.pkl'.format(model, dataset))

    # 根据tile_size_k和max_num_pattern构造文件名
    pattern_file = 'data/{}_{}_patterns_b{}k{}m{}_t.pkl'.format(model, dataset, batch_size, tile_size_k, max_num_pattern)
    
    if generate_pattern:
        nn_train = create_network(model, 'data/{}_{}_train_t.pkl'.format(model, dataset))
        patterns = defaultdict(dict)

        for layer in nn_train:
            if isinstance(layer, Conv2D):
                layer = conv2d_2_fc(layer)
            elif isinstance(layer, FC):
                pass
            else:
                continue

            tensor = layer.activation_tensor.sparse_map
            tensor = tensor.reshape(-1, tensor.shape[-1])

            for i in range(0, tensor.shape[1], tile_size_k):
                cur_tile_size_k = min(tile_size_k, tensor.shape[1] - i)
                tile = tensor[:, i:i+cur_tile_size_k]
                patterns[layer.name][i] = find_patterns_t(tile, max_num_pattern, cur_tile_size_k, T)

        # 将生成的patterns保存到文件中
        with open(pattern_file, 'wb') as f:
            pickle.dump(patterns, f)

        print("Patterns have been generated and saved.")

    else:
        # 从文件中加载patterns
        with open(pattern_file, 'rb') as f:
            patterns = pickle.load(f)
        
        print("Patterns have been loaded from the file.")

    for layer in nn_test:
        if isinstance(layer, Conv2D):
            layer = conv2d_2_fc(layer)
        elif isinstance(layer, FC):
            pass
        else:
            continue

        tensor = layer.activation_tensor.sparse_map
        tensor = tensor.reshape(-1, tensor.shape[-1])

        # prosperity_tensor, num_prefix = product_sparsify_whole_matrix(tensor, tile_size_k=16)
        # prosperity_cycles = torch.sum(prosperity_tensor).item() + num_prefix

        original_cycles = torch.sum(tensor).item()
        our_cycles_kmeanspp = 0
        optimal_cycles = 0

        for i in range(0, tensor.shape[1], tile_size_k):
            cur_tile_size_k = min(tile_size_k, tensor.shape[1] - i)
            tile = tensor[:, i:i+cur_tile_size_k]
            (cur_tile_cycles_kmeanspp, cur_tile_opt_cycles) = pattern_analysis_test(i, tile, patterns[layer.name][i])
            our_cycles_kmeanspp += cur_tile_cycles_kmeanspp
            optimal_cycles += cur_tile_opt_cycles

        # print("original cycles: ", original_cycles, "prosperity cycles: ", prosperity_cycles, "our cycles kmeanspp: ", our_cycles_kmeanspp, "optimal cycles: ", optimal_cycles)
        print("original cycles: ", original_cycles, "our cycles kmeanspp: ", our_cycles_kmeanspp, "optimal cycles: ", optimal_cycles)
            