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

def hag_pattern_analysis(tensor: torch.Tensor, max_num_pattern):
    M, K = tensor.shape
    capacity = max_num_pattern
    patterns = torch.zeros(capacity, K, dtype=torch.bool)
    # 计算初始边的数量
    V_A = set()
    V = set(range(K))
    # heap_dict = {}

    # 使用NumPy来计算邻接矩阵
    adj_matrix = np.zeros((max(M,K) + capacity, max(M,K) + capacity), dtype=bool)
    adj_matrix[:K, :M] = tensor.T

    def redundancy(v1: int, v2: int) -> int:
        return np.sum(adj_matrix[v1] & adj_matrix[v2])

    # 初始化堆和堆字典
    heap = []
    for v1 in range(K):
        for v2 in range(v1 + 1, K):
            r = redundancy(v1, v2)
            if r > 1:
                heapq.heappush(heap, (-r, (v1, v2)))
                # heap_dict[(v1, v2)] = -r

    while len(V_A) < capacity and heap:
        max_redundancy, (v1, v2) = heapq.heappop(heap)
        max_redundancy = -max_redundancy
        # del heap_dict[(v1, v2)]
        
        if max_redundancy > 1:
            w = max(M,K) + len(V_A)
            common_neighbors = adj_matrix[v1] & adj_matrix[v2]
            
            # 更新邻接矩阵
            adj_matrix[w] = common_neighbors
            # adj_matrix[v1] &= ~common_neighbors
            # adj_matrix[v2] &= ~common_neighbors
            # adj_matrix[v1, w] = True
            # adj_matrix[v2, w] = True

            if v2 < K:
                patterns[len(V_A), v1] = True
                patterns[len(V_A), v2] = True
            elif v1 < K:
                patterns[len(V_A), v1] = True
                patterns[len(V_A)] = patterns[len(V_A)] | patterns[v2 - max(M,K)]
            else:
                patterns[len(V_A)] = patterns[v1 - max(M,K)] | patterns[v2 - max(M,K)]
            
            V_A.add(w)

            # # 更新受影响的对的冗余度
            # for u in V.union(V_A) - {v1, v2, w}:               
            #     r = redundancy(w, u)
            #     if r > 1:
            #         heapq.heappush(heap, (-r, (u, w)))

             # 更新受影响的对的冗余度
            for u in V.union(V_A) - {v1, v2, w}:
                # for v in (v1, v2):
                #     smaller, larger = sorted((v, u))
                #     if (smaller, larger) in heap_dict:
                #         r = redundancy(smaller, larger)
                #         old_r = heap_dict[(smaller, larger)]
                #         heap.remove((old_r, (smaller, larger)))
                #         if r > 1:
                #             heapq.heappush(heap, (-r, (smaller, larger)))
                #             heap_dict[(smaller, larger)] = -r
                #         else:
                #             del heap_dict[(smaller, larger)]
                
                r = redundancy(w, u)
                if r > 1:
                    heapq.heappush(heap, (-r, (u,w)))
                    # heap_dict[(min(u, w), max(u, w))] = -r
            heapq.heapify(heap)
        else:
            break
    return patterns

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

def binary_weighted_k_means(pattern_dict: dict, k: int=256, tile_size: int=32):
    # filter out patterns with less than 2 nonzeros
    pattern_dict = {k: v for k, v in pattern_dict.items() if len(k) > 1}
    # sort the pattern according to value
    pattern_dict = dict(sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True))

    nodes = []
    for key in pattern_dict.keys():
        vec = torch.zeros(tile_size).to(torch.float32)
        for index in key:
            vec[index] = 1.0
        nodes.append(vec)
    weights = list(pattern_dict.values())


    # initialize centroids, a matrix
    centroids = torch.zeros(k, tile_size).to(torch.float32)
    # select top k patterns as centroids
    for i in range(k):
        if i < len(nodes):
            centroids[i] = nodes[i].clone()
        else:
            centroids[i] = torch.zeros(tile_size).to(torch.float32)

    # perform k-means
    for _ in range(10):
        # assign each node to the closest centroid
        assignments = []
        for node in nodes:
            # get hamming distance
            distances = torch.abs(centroids - node).sum(dim=-1)
            assignments.append(torch.argmin(distances))
        # update centroids with weighted average
        for i in range(k):
            indices = torch.tensor([j for j in range(len(nodes)) if assignments[j].item() == i])
            if len(indices) > 0:
                centroids[i] = torch.sum(torch.stack([weights[j] * nodes[j] for j in indices]), dim=0) / torch.sum(torch.tensor(weights)[indices])
    
    # round to the nearest integer
    centroids = [torch.round(centroid) for centroid in centroids]
    # convert to bool matrix
    centroids = torch.stack(centroids).to(torch.bool)
    return centroids

def find_patterns(tensor: torch.Tensor, max_num_pattern, cur_tile_size_k):

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

def pattern_analysis(tensor: torch.Tensor, max_num_pattern, cur_tile_size_k):

    # create a histogram of the number of nonzeros in each row
    # nnz_per_row = torch.sum(tensor != 0, dim=-1)
    # nnz_per_row = nnz_per_row.to(torch.float32).tolist()

    # print frequency of each number of nonzeros
    # freq = {}
    # for nnz in nnz_per_row:
    #     if nnz in freq:
    #         freq[nnz] += 1
    #     else:
    #         freq[nnz] = 1
    # print(freq)

    pattern_dict = defaultdict(int)
    for i in range(tensor.shape[0]):
        row = tensor[i]
        # convert row into nonzeros
        nonzeros = torch.nonzero(row).flatten()
        # convert to tuple
        nonzeros = tuple(nonzeros.tolist())
        pattern_dict[nonzeros] += 1
    
    # print("number of unique patterns: ", len(pattern_dict))
    # pattern_tensor_kmeans = binary_weighted_k_means(pattern_dict, k=max_num_pattern, tile_size=cur_tile_size_k)
    pattern_tensor_kmeanspp = binary_weighted_k_meansplusplus(pattern_dict, k=max_num_pattern, tile_size=cur_tile_size_k)
    # # sort according to value
    # sorted_pattern = sorted(pattern_dict.items(), key=lambda x: x[1], reverse=True)
    # selected_pattern = []
    # pattern_cnt = 0
    # for pattern in sorted_pattern:
    #     if (len(pattern[0]) > 1):
    #         selected_pattern.append(pattern)
    #         pattern_cnt += 1
    #         if pattern_cnt == max_num_pattern:
    #             break

    # # construct a matrix according to the selected pattern
    # pattern_tensor = torch.zeros(max_num_pattern, tensor.shape[1]).to(torch.bool)
    # for i, pattern in enumerate(selected_pattern):
    #     for index in pattern[0]:
    #         pattern_tensor[i, index] = True

    cycles_kmeans = 0
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

    return cycles_kmeans, cycles_kmeanspp, optimal_cycles

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

def plot_matrix(matrix: torch.Tensor, filename):

    # use seaborn plot
    cmap = plt.cm.colors.ListedColormap(["white", "#ADD8E6"])

    # Define the bounds and normalization for the colors
    bounds = [0, 0.5, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # save the matrix
    plt.imshow(matrix, cmap=cmap, norm=norm, aspect='equal')


    # Add a grid with thin lines
    plt.grid(which='major', color='black', linestyle='-', linewidth=0.5)

    # Turn off the axis labels
    plt.xticks([])
    plt.yticks([])
    # store as pdf and transparent
    plt.savefig('{}'.format(filename), transparent=True)
    return

def construct_subset_DAG(spike_tensor: torch.Tensor):
    argmax_size = 0
    shape = spike_tensor.shape
    spike_tensor = spike_tensor.reshape(-1, shape[-1])
    new_tensor = spike_tensor.clone()
    # create a graph with 16 nodes
    t = nx.DiGraph()
    t.add_nodes_from(range(spike_tensor.shape[0]))
    g = nx.DiGraph()
    g.add_nodes_from(range(spike_tensor.shape[0]))
    original_nnz = torch.sum(spike_tensor).item()
    rank_one_reduced_nnz = original_nnz
    rank_two_reduced_nnz = original_nnz
    ideal_reduced_nnz = original_nnz
    rank_one_util = 0
    rank_two_util = 0
    ideal_util = 0
    for m in range(spike_tensor.shape[0]):
        cur_row = spike_tensor[m]
        cur_nnz = torch.sum(cur_row).item()
        if cur_nnz < 2:
            continue
        and_result = torch.logical_and(cur_row, spike_tensor)
        equalities = torch.eq(and_result, spike_tensor)
        is_subset = torch.all(equalities, dim=-1)

        equalities = torch.eq(cur_row, spike_tensor)
        is_equal = torch.all(equalities, dim=-1)
        is_bigger_index = torch.arange(spike_tensor.shape[0]) >= m
        
        is_excluded = torch.logical_and(is_equal, is_bigger_index)
        is_real_subset = torch.logical_and(is_subset, ~is_excluded)

        argmax_size = max(argmax_size, torch.sum(is_real_subset).item())

        if torch.sum(is_real_subset) == 0:
            continue
        subset_index = torch.nonzero(is_real_subset).flatten()
        subset_size = torch.sum(spike_tensor[is_real_subset], dim=-1)
        subset_rows = spike_tensor[is_real_subset]
        max_subset_size, max_subset_index = torch.max(subset_size, dim=-1)
        
        if max_subset_size.item() < 1:
            continue
        t.add_edge(subset_index[max_subset_index].item(), m)
        new_tensor[m] = torch.logical_xor(new_tensor[m], spike_tensor[subset_index[max_subset_index]])
        rank_one_reduced_nnz -= max_subset_size.item()
        if torch.sum(is_real_subset).item() > 1:
            ideal_reduced_nnz -= cur_nnz
        else:
            ideal_reduced_nnz -= max_subset_size.item()
        rank_one_util += 1
        if max_subset_size.item() == cur_nnz or torch.sum(is_real_subset).item() == 1:
            ideal_util += 1
        else:
            ideal_util += 2

        maximum_size = max_subset_size.item()
        if max_subset_size.item() != cur_nnz:
            for i in range(subset_rows.shape[0]):
                if maximum_size == cur_nnz:
                    break
                cur_subset_row = subset_rows[i]
                if torch.sum(cur_subset_row) < 1:
                    continue
                # perform and with every row in the subset
                and_result = torch.logical_and(cur_subset_row, subset_rows)
                and_sum = torch.sum(and_result, dim=-1)
                for j in range(and_sum.shape[0]):
                    if and_sum[j].item() == 0:
                        maximum_size = max(maximum_size, torch.sum(cur_subset_row).item() + torch.sum(subset_rows[j]).item())

        
        rank_two_reduced_nnz -= maximum_size
        if maximum_size > max_subset_size.item():
            rank_two_util += 2
        else:
            rank_two_util += 1




        for i in subset_index:
            if spike_tensor[i].sum() < 1:
                continue
            g.add_edge(i.item(), m)

    rank_one_util = rank_one_util / spike_tensor.shape[0]
    rank_two_util = rank_two_util / (spike_tensor.shape[0] * 2)
    ideal_util = ideal_util / (spike_tensor.shape[0] * 2)
    total_elements = spike_tensor.shape[0] * spike_tensor.shape[1]


    # print("original_nnz", original_nnz, "rank_one_reduced_nnz: ", rank_one_reduced_nnz, "ideal_reduced_nnz: ", ideal_reduced_nnz)
    return rank_one_reduced_nnz, rank_two_reduced_nnz, ideal_reduced_nnz, original_nnz, total_elements, rank_one_util, rank_two_util, ideal_util

def visualize_dag(G, filename):
    # Check if the graph is a DAG
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The provided graph is not a Directed Acyclic Graph (DAG).")
    
    # Create position layout for the DAG
    pos = nx.topological_sort(G)
    pos = {node: (i, -len(list(nx.ancestors(G, node)))) for i, node in enumerate(pos)}
    
    # Draw the DAG
    nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=15, font_weight='bold', arrowstyle='-|>', arrowsize=15)
    plt.title("Directed Acyclic Graph (DAG) Visualization")
    plt.savefig('{}'.format(filename))

def visualize_matrix(matrix:torch.Tensor, filename):
    matrix = matrix.to(torch.int32)
    with open(filename, 'w') as f:
        for i in range(matrix.shape[0]):
            f.write(str(i).zfill(3) + ' ')
            for j in range(matrix.shape[1]):
                f.write(str(matrix[i, j].item()) + ' ')
            f.write('\n')

def visualize_matrix_with_parent(matrix:torch.Tensor, parents:torch.Tensor, filename):
    matrix = matrix.to(torch.int32)
    with open(filename, 'w') as f:
        for i in range(matrix.shape[0]):
            # make every i has same length
            f.write(str(i).zfill(3) + ' ')
            for j in range(matrix.shape[1]):
                f.write(str(matrix[i, j].item()) + ' ')
            f.write(str(parents[i].item()) + ' ')
            f.write('\n')


def children_count(tree, node):
    children = list(tree.successors(node))
    return len(children)

def buffering_analysis(nn):
    max_buffering_size = 0
    argmax_size = 0
    max_children = 0
    for ops in nn:
        if isinstance(ops, Conv2D):
            eq_fc = conv2d_2_fc(ops)
        elif isinstance(ops, FC):
            eq_fc = ops
        else:
            continue

        eq_fc.activation_tensor.sparse_map = eq_fc.activation_tensor.sparse_map.reshape(eq_fc.time_steps, eq_fc.sequence_length, eq_fc.input_dim)
        # transpose time step and sequence length
        eq_fc.activation_tensor.sparse_map = eq_fc.activation_tensor.sparse_map.permute(1, 0, 2).contiguous()
        input_shape = eq_fc.activation_tensor.shape
        input_shape = [np.prod(input_shape[:-1]), input_shape[-1]]
        input_tensor = eq_fc.activation_tensor.sparse_map.reshape(input_shape)

        tile_size_M = 256
        tile_size_N = 16

        
        for i in range(0, input_shape[0], tile_size_M):
            for j in range(0, input_shape[1], tile_size_N):
                tile = input_tensor[i:i+tile_size_M, j:j+tile_size_N]
                tree, graph, original_nnz, rank_one_reduced_nnz, ideal_reduced_nnz, cur_argmax_size = construct_subset_DAG(tile)

                argmax_size = max(argmax_size, cur_argmax_size)
                children_count_list = [children_count(tree, node) for node in tree.nodes]
                max_children = max(max_children, max(children_count_list))

                # print("max_children: ", max_children)


                # popcnt = tile.sum(dim=1)
                # # sort the popcnt
                # sorted_indices = torch.argsort(popcnt, descending=False)
                # # traverse the tree from the root
                # for node in nx.topological_sort(tree):
                #     tree.nodes[node]['num_children'] = len(list(tree.successors(node)))

                # cur_buffered_size = 0
                # # traverse in sorted indices
                # for i in sorted_indices:
                #     parent = list(tree.predecessors(i.item()))
                #     if len(parent) == 1:
                #         parent = parent[0]
                #         tree.nodes[parent]['num_children'] -= 1
                #         if tree.nodes[parent]['num_children'] == 0:
                #             cur_buffered_size -= 1
                #     elif len(parent) > 1:
                #         raise ValueError("The graph is not a tree")
                
                #     if tree.nodes[i.item()]['num_children'] != 0:
                #         cur_buffered_size += 1
                #         max_buffering_size = max(max_buffering_size, cur_buffered_size)

    print("argmax_size: ", argmax_size)
    print("max_buffering_size: ", max_buffering_size)
    print("max_children: ", max_children)

def idealistic_analysis():
    nn = create_network('spikformer', 'spikformer_cifar10.pkl')
    total_ori_nnz = 0
    total_rank_one_nnz = 0
    total_ideal_nnz = 0
    for ops in nn:
        if isinstance(ops, Conv2D):
            eq_fc = conv2d_2_fc(ops)
        elif isinstance(ops, FC):
            eq_fc = ops
        else:
            continue
        tensor = eq_fc.activation_tensor.sparse_map
        tensor = tensor.reshape(-1, tensor.shape[-1])
        shape = tensor.shape
        tile_size_M = 32
        tile_size_N = 16
        ori_nnzs = []
        rank_one_reduced_nnzs = []
        ideal_reduced_nnzs = []
        for i in range(0, shape[0], tile_size_M):
            for j in range(0, shape[1], tile_size_N):
                tile = tensor[i:i+tile_size_M, j:j+tile_size_N]
                tree, graph, original_nnz, rank_one_reduced_nnz, ideal_reduced_nnz = construct_subset_DAG(tile)
                visualize_dag(tree, 'tree.png')
                ori_nnzs.append(original_nnz)
                rank_one_reduced_nnzs.append(rank_one_reduced_nnz)
                ideal_reduced_nnzs.append(ideal_reduced_nnz)

        print("ori_nnzs: ", np.sum(ori_nnzs), "rank_one_reduced_nnzs: ", np.sum(rank_one_reduced_nnzs), "ideal_reduced_nnzs: ", np.sum(ideal_reduced_nnzs))
        rank_one_percentage = np.sum(rank_one_reduced_nnzs) / np.sum(ori_nnzs)
        ideal_percentage = np.sum(ideal_reduced_nnzs) / np.sum(ori_nnzs)
        print("rank_one_percentage: ", rank_one_percentage, "ideal_percentage: ", ideal_percentage)

        total_ori_nnz += np.sum(ori_nnzs)
        total_rank_one_nnz += np.sum(rank_one_reduced_nnzs)
        total_ideal_nnz += np.sum(ideal_reduced_nnzs)

    print("total_ori_nnz: ", total_ori_nnz, "total_rank_one_nnz: ", total_rank_one_nnz, "total_ideal_nnz: ", total_ideal_nnz)
    rank_one_percentage = total_rank_one_nnz / total_ori_nnz
    ideal_percentage = total_ideal_nnz / total_ori_nnz
    print("rank_one_percentage: ", rank_one_percentage, "ideal_percentage: ", ideal_percentage)

def nonzero_count(tensor: torch.Tensor, dim = 0):
    shape = tensor.shape
    tensor = tensor.reshape(shape[0], -1)
    nnz = torch.sum(tensor, dim=dim)
    nnz = nnz.to(torch.float32)
    # plot nnz into a histogram
    print("sparsity: ", get_density(tensor))
    print("mean: ", torch.mean(nnz).item(), "std: ", torch.std(nnz).item(), "max: ", torch.max(nnz).item(), "min: ", torch.min(nnz).item())

    return nnz

def whole_network_analysis(nn):
    nnz_list = []
    for op in nn:
        if isinstance(op, FC):
            tensor = op.activation_tensor.sparse_map
            nnz = nonzero_count(tensor, 0).tolist()
            nnz_list.extend(nnz)

    plt.hist(nnz, bins=4)
    plt.xlabel('Number of Nonzeros in this neuron')
    plt.ylabel('Frequency')
    plt.savefig('nnz_hist_neuron.png')

def all_zero_analysis():
    nn = create_network('spikformer', 'test.pkl')
    attn = nn[42]
    q = attn.act_q_tensor.sparse_map.reshape([attn.time_steps, attn.batch_size, attn.sequence_length, attn.num_head, attn.dim_per_head]).permute(0, 1, 3, 4, 2).contiguous()
    k = attn.act_k_tensor.sparse_map.reshape([attn.time_steps, attn.batch_size, attn.sequence_length, attn.num_head, attn.dim_per_head]).permute(0, 1, 3, 4, 2).contiguous()
    v = attn.act_v_tensor.sparse_map.reshape([attn.time_steps, attn.batch_size, attn.sequence_length, attn.num_head, attn.dim_per_head]).permute(0, 1, 3, 4, 2).contiguous()
    # reshape last dimension into 2 * 32
    q = q.reshape(q.shape[0], q.shape[1], q.shape[2], q.shape[3], 2, 32)
    k = k.reshape(k.shape[0], k.shape[1], k.shape[2], k.shape[3], 2, 32)
    v = v.reshape(v.shape[0], v.shape[1], v.shape[2], v.shape[3], 2, 32)
    q_row_nnz = torch.sum(q, dim=-1).permute(0, 1, 2, 4, 3)
    k_row_nnz = torch.sum(k, dim=-1).permute(0, 1, 2, 4, 3)
    v_row_nnz = torch.sum(v, dim=-1).permute(0, 1, 2, 4, 3)
    q_nonzero_row = torch.sum(q_row_nnz != 0, dim=-1)
    k_nonzero_row = torch.sum(k_row_nnz != 0, dim=-1)
    v_nonzero_row = torch.sum(v_row_nnz != 0, dim=-1)

    additional_overhead = attn.time_steps * attn.batch_size * attn.num_head * attn.dim_per_head * 2
    original_computation = attn.time_steps * attn.batch_size * attn.num_head * attn.dim_per_head * attn.dim_per_head * 2
    reduced_computation = torch.sum(k_nonzero_row * v_nonzero_row).item()

    print("sparsity of q: ", get_density(q), "sparsity of k: ", get_density(k), "sparsity of v: ", get_density(v))
    print("original_computation: ", original_computation, "reduced_computation: ", reduced_computation, "additional_overhead: ", additional_overhead)


def generate_random_matrix(m,k):
    tensor = torch.randint(0, 2, (m, k))
    return tensor

if __name__ == '__main__':
    
    model = 'spikformer'
    dataset_train = 'cifar10_train'
    dataset_test = 'cifar10_test'
    
    batch_size = 128
    tile_size_k = 16
    max_num_pattern = 256
    generate_pattern = False  # 设置为True以重新生成patterns

    nn_test = create_network(model, 'data/{}_{}.pkl'.format(model, dataset_test))

    # 根据tile_size_k和max_num_pattern构造文件名
    pattern_file = 'data/spikformer_cifar10_patterns_b{}k{}m{}.pkl'.format(batch_size, tile_size_k, max_num_pattern)
    
    if generate_pattern:
        nn_train = create_network(model, 'data/{}_{}.pkl'.format(model, dataset_train))
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
                patterns[layer.name][i] = find_patterns(tile, max_num_pattern, cur_tile_size_k)

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

        prosperity_tensor, num_prefix = product_sparsify_whole_matrix(tensor, tile_size_k=16)
        prosperity_cycles = torch.sum(prosperity_tensor).item() + num_prefix

        original_cycles = torch.sum(tensor).item()
        our_cycles_kmeanspp = 0
        optimal_cycles = 0

        for i in range(0, tensor.shape[1], tile_size_k):
            cur_tile_size_k = min(tile_size_k, tensor.shape[1] - i)
            tile = tensor[:, i:i+cur_tile_size_k]
            (cur_tile_cycles_kmeanspp, cur_tile_opt_cycles) = pattern_analysis_test(tile, patterns[layer.name][i])
            our_cycles_kmeanspp += cur_tile_cycles_kmeanspp
            optimal_cycles += cur_tile_opt_cycles

        print("original cycles: ", original_cycles, "prosperity cycles: ", prosperity_cycles, "our cycles kmeanspp: ", our_cycles_kmeanspp, "optimal cycles: ", optimal_cycles)
            