from networks import FC, Conv2D, MaxPool2D, LIFNeuron, Attention, create_network, conv2d_2_fc
import torch
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np
from utils import get_density, ceil_a_by_b

import seaborn as sns

def product_sparsify_whole_matrix(self, input_tensor: torch.Tensor):
    input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
    tile_size_k = 16
    tile_size_m = 256
    sparsified_tensor = torch.zeros_like(input_tensor)

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
                # if max_subset_size > 1: # can also reuse even when the size is 1
                sparsified_tile[i] = torch.logical_xor(sparsified_tile[i], max_subset)

            sparsified_tensor[m_id:m_id+cur_tile_size_m, k_id:k_id+cur_tile_siz_k] = sparsified_tile

        print("sparsity: ", get_density(sparsified_tensor))
        return sparsified_tensor

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
if __name__ == '__main__':

    model_list = ['lenet5_mnist', 'spikebert_sst2', 'spikebert_mr', 'spikebert_sst5', 
                     'spikingbert_sst2', 'spikingbert_qqp', 'spikingbert_mnli', 
                     'spikformer_cifar10', 'spikformer_cifar10dvs', 'spikformer_cifar100', 
                     'sdt_cifar10', 'sdt_cifar10dvs', 'sdt_cifar100',
                     'vgg16_cifar10', 'vgg16_cifar100', 
                       'resnet18_cifar10', 'resnet18_cifar100']
    
    model = 'spikformer'
    dataset = 'cifar100'

    nn = create_network(model, 'data/{}_{}.pkl'.format(model, dataset))
    max_density = 0
    max_density_id = 0
    id = 0
    for layer in nn:
        if isinstance(layer, Conv2D):
            eq_fc = conv2d_2_fc(layer)
            print("sparse: ", get_density(eq_fc.activation_tensor.sparse_map))
            if get_density(eq_fc.activation_tensor.sparse_map) > max_density:
                max_density = get_density(eq_fc.activation_tensor.sparse_map)
                max_density_id = id
        elif isinstance(layer, FC):
            print("sparse: ", get_density(layer.activation_tensor.sparse_map))
            if get_density(layer.activation_tensor.sparse_map) > max_density:
                max_density = get_density(layer.activation_tensor.sparse_map)
                max_density_id = id
        # elif isinstance(layer, FC):
        #     print("sparse: ", get_density(layer.activation_tensor.sparse_map))
        #     if get_density(layer.activation_tensor.sparse_map) > max_density:
        #         max_density = get_density(layer.activation_tensor.sparse_map)
        #         max_density_id = id
        id += 1

    layer = nn[max_density_id]
    if isinstance(layer, Conv2D):
        layer = conv2d_2_fc(layer)
    
    tensor = layer.activation_tensor.sparse_map
    tensor = tensor.reshape(-1, tensor.shape[-1])
    begin = 64
    end = 128
    plot_matrix(tensor[begin:end,begin:end], '{}_before.png'.format(model))
    print("before sparsity: ", get_density(tensor[begin:end,begin:end]))
    sparsified_tensor = product_sparsify_whole_matrix(layer, tensor)
    plot_matrix(sparsified_tensor[begin:end,begin:end], '{}_after.png'.format(model))
    print("after sparsity: ", get_density(sparsified_tensor[begin:end,begin:end]))

    raise ValueError("stop here")

    for model in model_list:
        model_name = model.split('_')[0]
        nn = create_network(model_name, 'data/{}.pkl'.format(model))

        tile_size_m = 256
        tile_size_k = 16
        rank_one_util_list = []
        rank_two_util_list = []
        ideal_util_list = []
        rank_one_nnz_list = []
        rank_two_nnz_list = []
        ideal_nnz_list = []
        original_nnz_list = []
        element_list = []
        all_output_dim = 0
        for op in nn:
            if isinstance(op, Conv2D):
                eq_fc = conv2d_2_fc(op)
            elif isinstance(op, FC):
                eq_fc = op
            else:
                continue
            tensor = eq_fc.activation_tensor.sparse_map
            tensor = tensor.reshape(-1, tensor.shape[-1])
            all_output_dim += eq_fc.output_dim
            print("model: ", model)
            print("sparsity: ", get_density(tensor))
            for i in range(0, tensor.shape[0], tile_size_m):
                for j in range(0, tensor.shape[1], tile_size_k):
                    cur_tile_size_m = min(tile_size_m, tensor.shape[0] - i)
                    cur_tile_siz_k = min(tile_size_k, tensor.shape[1] - j)
                    tile = tensor[i:i+cur_tile_size_m, j:j+cur_tile_siz_k]
                    rank_one_reduced_nnz, rank_two_reduced_nnz, ideal_reduced_nnz, original_nnz, total_elements, rank_one_util, rank_two_util, ideal_util = construct_subset_DAG(tile)
                    rank_one_util_list.append(rank_one_util * ceil_a_by_b(eq_fc.output_dim, 128))
                    ideal_util_list.append(ideal_util * ceil_a_by_b(eq_fc.output_dim, 128))
                    rank_one_nnz_list.append(rank_one_reduced_nnz * ceil_a_by_b(eq_fc.output_dim, 128))
                    ideal_nnz_list.append(ideal_reduced_nnz * ceil_a_by_b(eq_fc.output_dim, 128))
                    original_nnz_list.append(original_nnz * ceil_a_by_b(eq_fc.output_dim, 128))
                    rank_two_util_list.append(rank_two_util * ceil_a_by_b(eq_fc.output_dim, 128))
                    rank_two_nnz_list.append(rank_two_reduced_nnz * ceil_a_by_b(eq_fc.output_dim, 128))
                    element_list.append(total_elements * ceil_a_by_b(eq_fc.output_dim, 128))

        print("rank_one_util: ", np.mean(rank_one_util_list) / all_output_dim, "ideal_util: ", np.mean(ideal_util_list) / all_output_dim)
        print("rank_one_density: ", np.sum(rank_one_nnz_list) / np.sum(element_list), "ideal_density: ", np.sum(ideal_nnz_list) / np.sum(element_list))
        print("original_density: ", np.sum(original_nnz_list) / np.sum(element_list))
        print("rank_two_density: ", np.sum(rank_two_nnz_list) / np.sum(element_list), "rank_two_util: ", np.mean(rank_two_util_list) / all_output_dim)

        # write to file
        with open('rank12.txt'.format(model), 'a') as f:
            f.write("model: {}\n".format(model))
            f.write("rank_one_util: {}\n".format(np.sum(rank_one_util_list) / all_output_dim))
            f.write("ideal_util: {}\n".format(np.sum(ideal_util_list) / all_output_dim))
            f.write("rank_two_util: {}\n".format(np.sum(rank_two_util_list) / all_output_dim))
            f.write("rank_one_density: {}\n".format(np.sum(rank_one_nnz_list) / np.sum(element_list)))
            f.write("ideal_density: {}\n".format(np.sum(ideal_nnz_list) / np.sum(element_list)))
            f.write("original_density: {}\n".format(np.sum(original_nnz_list) / np.sum(element_list)))
            f.write("rank_two_density: {}\n".format(np.sum(rank_two_nnz_list) / np.sum(element_list)))


    # fc = nn[10]
    # mat = fc.activation_tensor.sparse_map
    # mat = mat.reshape(-1, mat.shape[-1])
    # mat = mat[0:256, 0:16]
    
    # t, g, original_nnz, rank_one_reduced_nnz, ideal_reduced_nnz, argmax_size, new_act = construct_subset_DAG(mat)
    # parents = torch.tensor([list(t.predecessors(node))[0] if len(list(t.predecessors(node))) > 0 else -1 for node in t.nodes])
    # visualize_matrix_with_parent(mat, parents, 'matrix.txt')
    # visualize_matrix(new_act, 'new_matrix.txt')
    # conv = nn[0]
    # whole_network_analysis(nn)

    # buffering_analysis(nn)
    # idealistic_analysis()