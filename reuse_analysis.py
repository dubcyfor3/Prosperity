from networks import FC, Conv2D, MaxPool2D, LIFNeuron, Attention, create_network, conv2d_2_fc
import torch
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np

def construct_subset_DAG(spike_tensor: torch.Tensor):
    shape = spike_tensor.shape
    spike_tensor = spike_tensor.reshape(-1, shape[-1])
    # create a graph with 16 nodes
    t = nx.DiGraph()
    t.add_nodes_from(range(spike_tensor.shape[0]))
    g = nx.DiGraph()
    g.add_nodes_from(range(spike_tensor.shape[0]))
    original_nnz = torch.sum(spike_tensor).item()
    rank_one_reduced_nnz = original_nnz
    ideal_reduced_nnz = original_nnz
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

        if torch.sum(is_real_subset) == 0:
            continue
        subset_index = torch.nonzero(is_real_subset).flatten()
        subset_size = torch.sum(spike_tensor[is_real_subset], dim=-1)
        max_subset_size, max_subset_index = torch.max(subset_size, dim=-1)
        
        if max_subset_size.item() >= 2:
            t.add_edge(subset_index[max_subset_index].item(), m)
            if cur_nnz == max_subset_size.item():
                rank_one_reduced_nnz -= max_subset_size.item() - 1
            else:
                rank_one_reduced_nnz -= max_subset_size.item()
            if cur_nnz - max_subset_size.item() == 0:
                ideal_reduced_nnz -= max_subset_size.item() - 1
            elif cur_nnz - max_subset_size.item() == 1:
                ideal_reduced_nnz -= max_subset_size.item()
            else:
                ideal_reduced_nnz -= cur_row.sum().item() - 2

        for i in subset_index:
            if spike_tensor[i].sum() < 2:
                continue
            g.add_edge(i.item(), m)

    # print("original_nnz", original_nnz, "rank_one_reduced_nnz: ", rank_one_reduced_nnz, "ideal_reduced_nnz: ", ideal_reduced_nnz)
    return t, g, original_nnz, rank_one_reduced_nnz, ideal_reduced_nnz

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

if __name__ == '__main__':
    nn = create_network('spikformer', 'test.pkl')
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
        tile_size_M = 256
        tile_size_N = 16
        ori_nnzs = []
        rank_one_reduced_nnzs = []
        ideal_reduced_nnzs = []
        for i in range(0, shape[0], tile_size_M):
            for j in range(0, shape[1], tile_size_N):
                tile = tensor[i:i+tile_size_M, j:j+tile_size_N]
                tree, graph, original_nnz, rank_one_reduced_nnz, ideal_reduced_nnz = construct_subset_DAG(tile)
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

