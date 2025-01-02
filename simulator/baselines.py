import torch
from utils import Stats, pad_to_power_of_2, img2col, ceil_a_by_b
from networks import Conv2D, FC, LIFNeuron, Attention, LayerNorm, conv2d_2_fc
from collections import OrderedDict
import numpy as np
from typing import Union
import prosparsity_engine


def run_simulation_sato(network):
    stats = OrderedDict()
    for operator in network:
        if isinstance(operator, Conv2D) or isinstance(operator, FC):
            stats[operator.name] = run_sato_conv_fc(operator)
        elif isinstance(operator, LIFNeuron):
            stats[operator.name] = run_sato_lif(operator)
    total_stats = Stats()
    for key, value in stats.items():
        total_stats += value
        total_stats.total_cycles += value.total_cycles
        
    print("total cycles: ", total_stats.total_cycles)
    print("total time: ", total_stats.total_cycles / (500 * 1000 * 1000))
    return total_stats

def run_simulation_eyeriss_linear(network):
    stats = OrderedDict()
    for operator in network:
        if isinstance(operator, Conv2D) or isinstance(operator, FC) or isinstance(operator, LIFNeuron):
            stats[operator.name] = run_eyeriss_conv_fc(operator)
        elif isinstance(operator, Attention):
            stats[operator.name] = run_eyeriss_attention(operator)
        elif isinstance(operator, LayerNorm):
            stats[operator.name] = run_eyeriss_layernorm(operator)
        else:
            continue
    
    total_stats = Stats()
    for key, value in stats.items():
        total_stats += value
        total_stats.total_cycles += value.total_cycles

    print("total cycles: ", total_stats.total_cycles)
    print("total time: ", total_stats.total_cycles / (500 * 1000 * 1000))
    return total_stats   

def run_simulation_eyeriss(network):
    stats = OrderedDict()
    for operator in network:
        if isinstance(operator, Conv2D) or isinstance(operator, FC) or isinstance(operator, LIFNeuron):
            stats[operator.name] = run_eyeriss_conv_fc(operator)
        else:
            continue
    
    total_stats = Stats()
    for key, value in stats.items():
        total_stats += value
        total_stats.total_cycles += value.total_cycles

    print("total cycles: ", total_stats.total_cycles)
    print("total time: ", total_stats.total_cycles / (500 * 1000 * 1000))
    return total_stats

def run_simulation_LoAS(network):
    stats = Stats()
    original_pseudo_acc = 0
    prosparsity_pseudo_acc = 0
    dense_acc = 0
    for operator in network:
        if isinstance(operator, Conv2D) or isinstance(operator, FC):
            ori_acc, prosparsity_acc, dense_acc = run_LoAS_convfc(operator)
    
            original_pseudo_acc += ori_acc
            prosparsity_pseudo_acc += prosparsity_acc

    bit_density = original_pseudo_acc / dense_acc
    product_density = prosparsity_pseudo_acc / dense_acc
    print("bit_density: ", bit_density)
    print("product_density: ", product_density)
    print("density improvement ratio: ", bit_density / product_density)
    return stats

def run_simulation_PTB(network):
    stats = OrderedDict()
    for operator in network:
        if isinstance(operator, Conv2D) or isinstance(operator, FC):
            stats[operator.name] = run_PTB_convfc(operator)
        elif isinstance(operator, LIFNeuron):
            stats[operator.name] = run_PTB_lif(operator)
    
    total_stats = Stats()
    for key, value in stats.items():
        total_stats += value
        total_stats.total_cycles += value.total_cycles

    print("total cycles: ", total_stats.total_cycles)
    print("total time: ", total_stats.total_cycles / (500 * 1000 * 1000))
    return total_stats

def run_simulation_MINT(network):
    stats = OrderedDict()
    for operator in network:
        if isinstance(operator, Conv2D) or isinstance(operator, FC):
            stats[operator.name] = run_MINT_convfc(operator)
        elif isinstance(operator, LIFNeuron):
            stats[operator.name] = run_PTB_lif(operator)
    
    total_stats = Stats()
    for key, value in stats.items():
        total_stats += value
        total_stats.total_cycles += value.total_cycles

    print("total cycles: ", total_stats.total_cycles)
    print("total time: ", total_stats.total_cycles / (500 * 1000 * 1000))
    return total_stats


def run_LoAS_convfc(operator: Union[FC, Conv2D]):

    if isinstance(operator, Conv2D):
        eq_fc = conv2d_2_fc(operator)
    elif isinstance(operator, FC):
        eq_fc = operator

    input_tensor = eq_fc.activation_tensor.sparse_map.clone()

    input_tensor = input_tensor.reshape(eq_fc.time_steps, eq_fc.sequence_length, eq_fc.input_dim)
    # pad the time step into power of 2
    input_tensor = pad_to_power_of_2(input_tensor, 0)


    time_step_group = 4

    input_tensor = input_tensor.reshape(time_step_group, -1, eq_fc.sequence_length, eq_fc.input_dim)
    input_tensor = input_tensor.sum(dim=1).to(torch.bool)
    # swap the first two dimension
    input_tensor = input_tensor.permute(1, 0, 2).contiguous()
    input_tensor = input_tensor.reshape(-1, eq_fc.input_dim)

    prosparsity_act, prefix = prosparsity_engine.find_product_sparsity(input_tensor, 256, 16)

    original_pseudo_acc = torch.sum(input_tensor).item() * eq_fc.output_dim
    prosparsity_pseudo_acc = torch.sum(prosparsity_act).item() * eq_fc.output_dim
    dense_acc = input_tensor.shape[0] * input_tensor.shape[1] * eq_fc.output_dim

    return original_pseudo_acc, prosparsity_pseudo_acc, dense_acc


def run_MINT_convfc(operator: Union[FC, Conv2D]):

    stats = Stats()
    if isinstance(operator, Conv2D):
        eq_fc = conv2d_2_fc(operator)
    elif isinstance(operator, FC):
        eq_fc = operator
    else:
        raise Exception("unsupported operator")
    
    num_PE = 128

    input_tensor = eq_fc.activation_tensor.sparse_map.clone()
    input_tensor = input_tensor.reshape(eq_fc.time_steps, eq_fc.sequence_length, eq_fc.input_dim)

    compute_cycles = input_tensor.sum().item() * ceil_a_by_b(eq_fc.output_dim, num_PE)

    stats.compute_cycles = compute_cycles
    stats.total_cycles = compute_cycles

    print(operator.name)
    print("total cycles: ", stats.total_cycles)

    return stats

def StSAP(act: torch.Tensor):
    # systolic array of size 16x8
    whole_act = act.clone()
    # processed_act = whole_act.reshape(-1, act.shape[-1])
    total_length = 0
    for i in range(whole_act.shape[0]):
        processed_act = whole_act[i]
        # find row with all zero
        all_zero_row = torch.sum(processed_act != 0, dim=-1) == 0
        # find row with all nonzero
        all_nonzero_row = torch.sum(processed_act != 0, dim=-1) == processed_act.shape[-1]
        excepted_row = torch.logical_or(all_zero_row, all_nonzero_row)
        # get all the row except all zero row and all nonzero row
        processed_act = processed_act[~excepted_row]
        cur_size = processed_act.shape[0]
        searched_row = torch.zeros(processed_act.shape[0], dtype=torch.bool)
        for i in range(processed_act.shape[0] - 1):
            if searched_row[i]:
                continue
            cur_row = processed_act[i]
            rest_row = processed_act[i+1:]
            and_result = torch.logical_and(cur_row, rest_row)
            non_overlap_row = torch.sum(and_result, dim=-1) == 0
            # pad the left of non_overlap_row to the original shape
            non_overlap_row = torch.cat([torch.zeros(i + 1, dtype=torch.bool), non_overlap_row])
            non_overlap_row = torch.logical_and(non_overlap_row, ~searched_row)
            # find the first nonzero in non_overlap_row
            first_nonzero = torch.argmax(non_overlap_row.to(torch.int)).item()
            if first_nonzero != 0:
                cur_size -= 1
                searched_row[first_nonzero] = True
        processed_size = cur_size + torch.sum(all_nonzero_row)
        total_length += processed_size.item()
    return total_length

def run_eyeriss_conv_fc(operator: Union[FC, Conv2D, LIFNeuron]):
    # a 14 * 12 systolic array, weight stationary
    stats = Stats()
    if isinstance(operator, FC):
        stats.compute_cycles += ceil_a_by_b(operator.output_dim, 12) * (operator.input_dim * operator.batch_size * operator.time_steps * operator.sequence_length // 14)
    elif isinstance(operator, Conv2D):
        eq_fc = conv2d_2_fc(operator)
        stats.compute_cycles += ceil_a_by_b(eq_fc.output_dim, 12) * (eq_fc.input_dim * eq_fc.batch_size * eq_fc.time_steps * eq_fc.sequence_length // 14)
    elif isinstance(operator, LIFNeuron):
        stats.compute_cycles += (operator.time_steps // 12) * operator.num_neuron * operator.batch_size // 14
    else:
        pass
    stats.total_cycles = stats.compute_cycles
    print(operator.name)
    print("total cycles: ", stats.total_cycles)
    return stats

def run_eyeriss_layernorm(operator: LayerNorm):
    stats = Stats()
    num_ops = operator.activation_tensor.get_size() + (operator.activation_tensor.get_size() // operator.dim) * 2 + operator.activation_tensor.get_size() + operator.activation_tensor.get_size()
    stats.compute_cycles += num_ops // 168
    stats.total_cycles = stats.compute_cycles
    print(operator.name)
    print("total cycles: ", stats.total_cycles)
    return stats

def run_eyeriss_attention(operator: Attention):
    stats = Stats()
    if operator.attention_type == 'spikformer' or operator.attention_type == 'spikebert':
        num_ops = operator.sequence_length * operator.sequence_length * operator.batch_size * operator.num_head * operator.dim_per_head * operator.time_steps * 2
        stats.compute_cycles += num_ops // 168
    elif operator.attention_type == 'sdt':
        num_ops = operator.sequence_length * operator.dim_per_head * operator.batch_size * operator.time_steps * operator.num_head * 4
        stats.compute_cycles += num_ops // 168
    elif operator.attention_type == 'spikingbert':
        num_ops = operator.sequence_length * operator.sequence_length * operator.batch_size * operator.num_head * operator.dim_per_head * operator.time_steps * 2
        num_ops += operator.sequence_length * operator.sequence_length * operator.batch_size * operator.num_head * operator.time_steps * 2 + operator.sequence_length * operator.batch_size * operator.num_head * operator.time_steps
        stats.compute_cycles += num_ops // 168
    else:
        raise Exception("unsupported attention type")
    
    stats.total_cycles = stats.compute_cycles
    return stats
    

def run_sato_conv_fc(operator: Union[FC, Conv2D]):
    # 128 PE
    stats = Stats()
    if isinstance(operator, FC):
        # do nothing
        eq_fc = operator
        pass
    elif isinstance(operator, Conv2D):
        eq_fc = conv2d_2_fc(operator)
    input_tensor = eq_fc.activation_tensor.sparse_map.reshape(-1, eq_fc.activation_tensor.sparse_map.shape[-1])
    nnz_each_row = torch.sum(input_tensor != 0, dim=-1)
    PE_spikes = torch.zeros(128, dtype=torch.int)
    for i in range(input_tensor.shape[0]):
        min_idx = torch.argmin(PE_spikes)
        PE_spikes[min_idx] += nnz_each_row[i].item()
    stats.compute_cycles = torch.max(PE_spikes).item() * eq_fc.output_dim
    stats.total_cycles = stats.compute_cycles
    print(operator.name)
    print("total cycles: ", stats.compute_cycles)
    return stats

def run_sato_lif(operator: LIFNeuron):
    stats = Stats()
    stats.compute_cycles = operator.num_neuron * (torch.log2(torch.tensor(operator.time_steps)).item() + 1) # 1 for leaf node comparator
    stats.total_cycles = stats.compute_cycles
    return stats

def run_PTB_convfc(operator: Union[FC, Conv2D]):
    # a 16 x 8 systolic array or 21 x 8
    stats = Stats()
    time_window_size = 4
    # pad the time step into power of 2
    operator.activation_tensor.sparse_map = pad_to_power_of_2(operator.activation_tensor.sparse_map, 0)
    input_shape = operator.activation_tensor.sparse_map.shape
    input_shape = [dim for dim in input_shape if dim != 1]
    new_shape = [-1, time_window_size]
    new_shape.extend(input_shape[1:])
    input_tensor = operator.activation_tensor.sparse_map.reshape(new_shape)
    input_tensor = input_tensor.sum(dim=1)
    if isinstance(operator, FC):
        if len(input_tensor.shape) == 2:
            # may not fully utilize systolic array if the first dimension is not 8
            input_tensor = input_tensor.permute(1, 0).contiguous()
        elif len(input_tensor.shape) == 3:
            # the first dimension should be 8 to fully utilize the systolic array, change the second dimension to make first dimension 8
            input_tensor = input_tensor.reshape([8, -1, input_tensor.shape[-1]])
            input_tensor = input_tensor.permute(1, 2, 0).contiguous()
        else:
            raise Exception("unsupported input shape")
        sequence_length = operator.sequence_length
        input_dim = operator.input_dim
        output_dim = operator.output_dim
    elif isinstance(operator, Conv2D):
        input_tensor = img2col(input_tensor, operator.kernel_size, operator.stride, operator.padding)
        # the first dimension should be 8 to fully utilize the systolic array, change the second dimension to make first dimension 8
        input_tensor = input_tensor.reshape([8, -1, input_tensor.shape[-1]])
        input_tensor = input_tensor.permute(1, 2, 0).contiguous()
        sequence_length = operator.output_H * operator.output_H
        input_dim = operator.input_channel * operator.kernel_size * operator.kernel_size
        output_dim = operator.output_channel
    if True:
        input_length = StSAP(input_tensor)
    else:
        input_length = np.prod(input_tensor.shape[:-1])
    repeate_times = ceil_a_by_b(output_dim, 16)
    stats.compute_cycles += input_length * repeate_times * (time_window_size) # one stage for leak and one stage for spike generate
    num_cold_start = sequence_length
    stats.compute_cycles += num_cold_start * 16 * 2

    stats.reads['dram'] += operator.activation_tensor.get_size()
    stats.reads['dram'] += operator.weight_tensor.get_size()
    stats.writes['dram'] += operator.output_tensor.get_size() // 8
    init_mem_access = 16 * 8 * (8 + 4)
    total_mem_access = stats.reads['dram'] + stats.writes['dram']
    middle_mem_access = total_mem_access - init_mem_access
    init_latency = init_mem_access // 1024
    middle_latency = middle_mem_access // 1024
    stats.mem_stall_cycles = init_latency + max(0, middle_latency - stats.compute_cycles)
    stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles

    print(operator.name)
    print("mem stall: ", stats.mem_stall_cycles)
    print("total cycles: ", stats.total_cycles)
    return stats

def run_PTB_lif(operator: LIFNeuron):
    stats = Stats()
    stats.compute_cycles = operator.num_neuron * operator.time_steps * 3 // 16
    stats.total_cycles = stats.compute_cycles
    print(operator.name)
    print("total cycles: ", stats.compute_cycles)
    return stats

def get_stats_stellar(name: str):
    time_dict = {
        'vgg16_cifar10': 0.040187906,
        'vgg16_cifar100': 0.077698212,
    }

    stats = Stats()
    if name not in time_dict:
        stats.total_cycles = None
    else:
        stats.total_cycles = time_dict[name] * 500 * 1000 * 1000
    return stats

def get_stats_A100(name: str):
    time_dict = {
        'spikformer_cifar10': 0.019,
        'spikformer_cifar10dvs': 0.0114,
        'spikformer_cifar100': 0.019,
        'sdt_cifar10': 0.017,
        'sdt_cifar10dvs': 0.018,
        'sdt_cifar100': 0.018,
        'spikebert_sst2': 0.0372,
        'spikebert_mr': 0.0397,
        'spikebert_sst5': 0.033357901,
        'spikingbert_sst2': 0.096202657,
        'spikingbert_qqp': 0.092028516,
        'spikingbert_mnli': 0.094227655,

    }
    stats = Stats()
    if name not in time_dict:
        stats.total_cycles = None
    else:
        stats.total_cycles = time_dict[name] * 500 * 1000 * 1000
    return stats
