from networks import FC, Conv2D, MaxPool2D, LIFNeuron, Attention, create_network, conv2d_2_fc
import torch
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from utils import get_density

def construct_subset_DAG(spike_tensor: torch.Tensor):
    argmax_size = 0
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

        argmax_size = max(argmax_size, torch.sum(is_real_subset).item())

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
    return t, g, original_nnz, rank_one_reduced_nnz, ideal_reduced_nnz, argmax_size

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

def buffering_analysis(nn):
    max_buffering_size = 0
    argmax_size = 0
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
if __name__ == '__main__':
    # nn = create_network('spikformer', 'spikformer_cifar10.pkl')
    # fc = nn[10]
    # conv = nn[0]
    # whole_network_analysis(nn)
    buffering_analysis()

    # idealistic_analysis()