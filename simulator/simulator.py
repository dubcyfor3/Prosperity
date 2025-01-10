from networks import FC, Conv2D, MaxPool2D, LIFNeuron, Attention, LayerNorm, create_network, conv2d_2_fc, extract_kernel_size
from utils import ceil_a_by_b, Stats, write_position, write_title
import torch
import numpy as np
from collections import defaultdict
from collections import OrderedDict
import logging
from typing import Union
import argparse
import os
import networkx as nx
from accelerator import Accelerator
from baselines import run_simulation_eyeriss, run_simulation_eyeriss_linear, run_simulation_PTB, run_simulation_sato, run_simulation_MINT, run_simulation_LoAS, get_stats_stellar, get_stats_A100
from energy import get_total_energy
import prosparsity_engine

logging.basicConfig(level=logging.INFO)



ori_nnzs = []
processed_nnzs = []
rank_one_prefix = []
ops_nnzs = []
rank_two_nnzs = []
rank_two_prefix = []
total_elements = []
num_all_zero_row = []
num_EM_row = []
num_PM_row = []
num_other_row = []
argmax_entries = 0

def clear_global_stats():
    global ori_nnzs
    global processed_nnzs
    global total_elements
    global ops_nnzs
    global rank_two_nnzs
    global rank_one_prefix
    global rank_two_prefix
    global num_all_zero_row
    global num_EM_row
    global num_PM_row
    global num_other_row

    ori_nnzs = []
    processed_nnzs = []
    total_elements = []
    ops_nnzs = []
    rank_two_nnzs = []
    rank_one_prefix = []
    rank_two_prefix = []
    num_all_zero_row = []
    num_EM_row = []
    num_PM_row = []
    num_other_row = []

        


class Simulator:
    def __init__(self, accelerator: Accelerator, network: list, benchmark_name, use_cuda=False):
        self.accelerator = accelerator
        self.benchmark_name = benchmark_name
        self.network = network
        self.track_sparsity_increment = True
        self.test_rank_two = False

        # check whether cuda is available
        if use_cuda:
            assert torch.cuda.is_available()
        self.device = 'cuda' if use_cuda else 'cpu'

    def sim(self):
        if self.accelerator.type == 'PTB':
            stats = run_simulation_PTB(self.network)
        elif self.accelerator.type == 'Eyeriss':
            stats = run_simulation_eyeriss(self.network)
        elif self.accelerator.type == 'Eyeriss_linear':
            stats = run_simulation_eyeriss_linear(self.network)
        elif self.accelerator.type == 'SATO':
            stats = run_simulation_sato(self.network)
        elif self.accelerator.type == 'MINT':
            stats = run_simulation_MINT(self.network)
        elif self.accelerator.type == 'LoAS':
            stats = run_simulation_LoAS(self.network)
        elif self.accelerator.type == 'Stellar':
            stats = get_stats_stellar(self.benchmark_name)
        elif self.accelerator.type == 'A100':
            stats = get_stats_A100(self.benchmark_name)
        elif self.accelerator.type == 'Prosperity':
            stats = self.run_simulation()
        else:
            raise Exception("unsupported accelerator type")

        return stats
    

    
    def run_simulation(self, linear_workload=False):
        stats = OrderedDict()
        spike_stored_in_buffer = False
        for operator in self.network:
            if isinstance(operator, FC):
                if self.device == 'cuda':
                    stats[operator.name] = self.run_fc_cuda(operator, spike_stored_in_buffer)
                else:
                    stats[operator.name] = self.run_fc(operator, spike_stored_in_buffer)
            elif isinstance(operator, Conv2D):
                if self.device == 'cuda':
                    stats[operator.name] = self.run_conv2d_cuda(operator, spike_stored_in_buffer)
                else:
                    stats[operator.name] = self.run_conv2d(operator, spike_stored_in_buffer)
            elif isinstance(operator, LIFNeuron):
                stats[operator.name], spike_stored_in_buffer = self.run_LIF(operator)
            elif isinstance(operator, Attention):
                if linear_workload:
                    continue
                stats[operator.name], spike_stored_in_buffer = self.run_attention(operator, spike_stored_in_buffer)
            elif isinstance(operator, MaxPool2D):
                stats[operator.name] = self.run_maxpool(operator)
            elif isinstance(operator, LayerNorm):
                if linear_workload:
                    continue
                stats[operator.name] = self.run_layernorm(operator)
            else:
                raise Exception("unsupported operator")

        total_stats = Stats()
        last_stats_key, last_stats = next(iter(stats.items()))
        cycles_layer = defaultdict(int)
        for key, value in stats.items():
            total_stats += value
            type = key.split('_')[0]
            # lif can be overlapped with fc and conv
            if key.startswith('lif') and (last_stats_key.startswith('fc') or last_stats_key.startswith('conv') or last_stats_key.startswith('layernorm')):
                total_stats.total_cycles += value.LIF_latency
                cycles_layer[type] += value.LIF_latency
                # assume time of fc and conv is larger than lif

            elif key.startswith('lif') and last_stats_key.startswith('attention'):
                if last_stats.total_cycles + value.LIF_latency >= value.total_cycles:
                    total_stats.total_cycles += value.LIF_latency
                    cycles_layer[type] += value.LIF_latency
                else:
                    total_stats.total_cycles += value.total_cycles - last_stats.total_cycles
                    cycles_layer[type] += value.total_cycles - last_stats.total_cycles
            else:
                total_stats.total_cycles += value.total_cycles
                cycles_layer[type] += value.total_cycles
            last_stats_key, last_stats = key, value

        
        print("total cycles: ", total_stats.total_cycles)
        print("time", total_stats.total_cycles / (500 * 1000 * 1000))
        print("total ops: ", total_stats.num_ops)
        print("mem access", total_stats.reads['dram'] + total_stats.writes['dram'])
        print("buffer access", total_stats.reads['g_act'] + total_stats.writes['g_act'] + total_stats.reads['g_wgt'] + total_stats.writes['g_wgt'] + total_stats.reads['g_psum'] + total_stats.writes['g_psum'])
        print("preprocess stall cycles: ", total_stats.preprocess_stall_cycles)
        print("mem stall cycles: ", total_stats.mem_stall_cycles)
 
        if self.accelerator.product_sparsity and self.track_sparsity_increment:
            bit_density = sum(ori_nnzs) / sum(total_elements)
            product_density = sum(processed_nnzs) / sum(total_elements)

            total_stats.bit_density = bit_density
            total_stats.product_density = product_density
            print("bit density: ", bit_density)
            print("prosparsity density: ", product_density)
            if self.test_rank_two:
                # ops_sparsity = sum(ops_nnzs) / sum(total_elements)
                # rank_two_sparsity = sum(rank_two_nnzs) / sum(total_elements)
                # total_stats.ops_sparsity = ops_sparsity
                # total_stats.rank_two_sparsity = rank_two_sparsity
                avg_rank_one_prefix = sum(rank_one_prefix) / len(rank_one_prefix)
                avg_rank_two_prefix = sum(rank_two_prefix) / len(rank_two_prefix)
                total_stats.avg_rank_one_prefix = avg_rank_one_prefix
                total_stats.avg_rank_two_prefix = avg_rank_two_prefix

                # print("ops sparsity: ", ops_sparsity)
                # print("rank two sparsity: ", rank_two_sparsity)

        # total_stats.cycle_breakdown = cycles_layer
        # for key, value in cycles_layer.items():
        #     print(f"{key}: {value / total_stats.total_cycles}")
        

        return total_stats
    
    def run_fc_cuda(self, operator: FC, spike_stored_in_buffer=False, weight_stored_in_buffer=False):
        stats = Stats()
        assert operator.activation_tensor.shape[-1] == operator.weight_tensor.shape[0]
        # only deal with batch size 1   
        assert operator.time_steps * operator.sequence_length * operator.input_dim == operator.activation_tensor.sparse_map.numel()
        # reshape activation tensor
        operator.activation_tensor.sparse_map = operator.activation_tensor.sparse_map.reshape(operator.time_steps, operator.sequence_length, operator.input_dim)
        operator.activation_tensor.sparse_map = operator.activation_tensor.sparse_map.permute(1, 0, 2).contiguous()

        input_shape = operator.activation_tensor.shape
        input_shape = [np.prod(input_shape[:-1]), input_shape[-1]]
        input_act = operator.activation_tensor.sparse_map.reshape(input_shape)
        M, K, N = input_shape[0], input_shape[1], operator.weight_tensor.shape[1]

        tile_size_N = self.accelerator.adder_array_size
        tile_size_M = self.accelerator.SpMM_tile_size_M
        tile_size_K = self.accelerator.SpMM_tile_size_K

        tile_num_M = ceil_a_by_b(M, tile_size_M)
        tile_num_K = ceil_a_by_b(K, tile_size_K)
        tile_num_N = ceil_a_by_b(N, tile_size_N)

        if tile_size_M * tile_size_K > 32768:
            raise Exception("tile size is too large for shared memory")
        prosparsity_act, prefix_array = prosparsity_engine.find_product_sparsity(input_act, tile_size_M, tile_size_K)
        kernel_size = extract_kernel_size(operator.name)

        stats.reads['dram'] =  K * N * operator.weight_tensor.nbits * tile_num_M
        stats.reads['dram'] += M * K * operator.activation_tensor.nbits * tile_num_N // (kernel_size * kernel_size)
        stats.writes['g_act'] = M * K * operator.activation_tensor.nbits * tile_num_N
        stats.writes['g_wgt'] = K * N * operator.weight_tensor.nbits * tile_num_M

        stats.reads['g_act'] = M * K * operator.activation_tensor.nbits * tile_num_N
        stats.reads['g_wgt'] = torch.sum(prosparsity_act != 0).item() * N * operator.weight_tensor.nbits * tile_num_M
        stats.writes['g_psum'] = M * N * operator.output_tensor.nbits * tile_num_K

        # pad the input_act and prosparsity_act to the multiple of tile_size_K
        pad_size = ceil_a_by_b(K, tile_size_K) * tile_size_K - K
        input_act = torch.cat([input_act, torch.zeros(M, pad_size).to(input_act.device)], dim=-1)
        prosparsity_act = torch.cat([prosparsity_act, torch.zeros(M, pad_size).to(prosparsity_act.device)], dim=-1)


        prosparsity_act = prosparsity_act.reshape(-1, tile_size_K)
        input_act = input_act.reshape(-1, tile_size_K)

        # get all zero row
        nnz_each_row_ori = torch.sum(input_act != 0, dim=-1)
        all_zero_row_ori = nnz_each_row_ori == 0
        nnz_each_row = torch.sum(prosparsity_act != 0, dim=-1)
        all_zero_row = nnz_each_row == 0

        compute_cycles = (torch.sum(prosparsity_act != 0).item() + torch.sum(all_zero_row).item() - torch.sum(all_zero_row_ori).item()) * tile_num_N
        preprocess_cycles = (get_prosparsity_cycles(input_act) + M // self.accelerator.num_popcnt) * tile_num_N

        if self.accelerator.issue_type == 1:
            compute_cycles = 0
            for i in range(prefix_array.shape[0]):
                for j in range(prefix_array.shape[1]):
                    cur_start_row = i * prefix_array.shape[0] * tile_size_M
                    cur_end_row = min((i + 1) * prefix_array.shape[0] * tile_size_M, M)
                    cur_prosparsity_act = prosparsity_act[cur_start_row: cur_end_row, :]
                    cur_input_act = input_act[cur_start_row: cur_end_row, :]
                    all_zero_row = torch.sum(cur_prosparsity_act != 0, dim=-1) == 0
                    all_zero_row_ori = torch.sum(cur_input_act != 0, dim=-1) == 0
                    cur_compute_cycles = (torch.sum(cur_prosparsity_act != 0).item() + torch.sum(all_zero_row).item() - torch.sum(all_zero_row_ori).item()) * tile_num_N
                    cur_prefix = prefix_array[i, j, :]
                    cur_forest = construct_prosparsity_forest(cur_prefix)
                    depth = nx.dag_longest_path_length(cur_forest)
                    cur_issue_cycles = depth // 4 * tile_size_M * tile_num_N
                    compute_cycles += max(cur_issue_cycles, cur_compute_cycles)
        elif self.accelerator.issue_type == 2:
            pass
        else:
            raise Exception("unsupported issue type")

        init_mem_access = 0
        if not weight_stored_in_buffer:
            init_mem_access += min(tile_size_K, K) * min(tile_size_N, N) * operator.weight_tensor.nbits # read the first tile from dram to buffer
        if not spike_stored_in_buffer:
            init_mem_access += min(tile_size_K, K) * min(tile_size_M, M)

        total_mem_access = stats.reads['dram'] + stats.writes['dram']
        middle_mem_access = total_mem_access - init_mem_access
        init_latency = init_mem_access // self.accelerator.mem_if_width
        middle_latency = middle_mem_access // self.accelerator.mem_if_width
        stats.compute_cycles = max(compute_cycles, preprocess_cycles)
        stats.preprocess_stall_cycles = max(0, preprocess_cycles - compute_cycles)
        stats.mem_stall_cycles = init_latency + max(0, middle_latency - stats.compute_cycles)
        stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles

        ori_nnzs.append(torch.sum(input_act != 0).item() * tile_num_N) 
        processed_nnzs.append(torch.sum(prosparsity_act != 0).item() * tile_num_N)
        total_elements.append(M * K * tile_num_N)

        print(operator.name)
        print("compute cycles: ", compute_cycles)

        return stats


    def run_fc(self, operator: FC, spike_stored_in_buffer=False, weight_stored_in_buffer=False):
        stats = Stats()
        assert operator.activation_tensor.shape[-1] == operator.weight_tensor.shape[0]
        # only deal with batch size 1
        assert operator.time_steps * operator.sequence_length * operator.input_dim == operator.activation_tensor.sparse_map.numel()
        # reshape activation tensor
        # to device
        operator.activation_tensor.sparse_map = operator.activation_tensor.sparse_map
        operator.activation_tensor.sparse_map = operator.activation_tensor.sparse_map.reshape(operator.time_steps, operator.sequence_length, operator.input_dim)
        # transpose time step and sequence length
        operator.activation_tensor.sparse_map = operator.activation_tensor.sparse_map.permute(1, 0, 2).contiguous()


        input_shape = operator.activation_tensor.shape
        input_shape = [np.prod(input_shape[:-1]), input_shape[-1]]
        
        input_tensor = operator.activation_tensor.sparse_map.reshape(input_shape)
        M, K, N = input_shape[0], input_shape[1], operator.weight_tensor.shape[1]
        tile_size_M = self.accelerator.SpMM_tile_size_M
        tile_size_K = self.accelerator.SpMM_tile_size_K
        tile_size_N = self.accelerator.adder_array_size
        tile_num_M = ceil_a_by_b(M, tile_size_M)
        tile_num_K = ceil_a_by_b(K, tile_size_K)
        tile_num_N = ceil_a_by_b(N, tile_size_N)

        act_tile_size = tile_size_M * tile_size_K * 1 # spike is one bit
        wgt_tile_size = tile_size_K * tile_size_N * operator.weight_tensor.nbits

        if M * K <= self.accelerator.sram_size['act']:
            buffer_state_act = "store all"
        elif act_tile_size * tile_num_K <= self.accelerator.sram_size['act']:
            buffer_state_act = "store row"
        elif act_tile_size <= self.accelerator.sram_size['act']:
            buffer_state_act = "store single tile"
        else:
            raise Exception("single tile cannot fit in sram act buffer")
        
        if K * N * operator.weight_tensor.nbits <= self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store all"
        elif wgt_tile_size * tile_num_K <= self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store col"
        elif wgt_tile_size <= self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store single tile"
        else:
            raise Exception("single tile cannot fit in sram wgt buffer")

        # 1. load activation and weight to sram
        # 2. load activation and weight to local buffer
        # 3. do preprocess to the activation
        # 4. start computation
        compute_cycles = 0
        preprocess_cycles = 0
        orginial_compute_cycles = 0

        is_conv = operator.name.endswith('_fc')
        kernel_size = extract_kernel_size(operator.name)


        for m in range(tile_num_M):
            for n in range(tile_num_N):
                for k in range(tile_num_K):
                    cur_tile_size_M = min(tile_size_M, M - m * tile_size_M)
                    cur_tile_size_K = min(tile_size_K, K - k * tile_size_K)
                    cur_tile_size_N = min(tile_size_N, N - n * tile_size_N)
                    cur_act = input_tensor[m * tile_size_M: m * tile_size_M + cur_tile_size_M, k * tile_size_K: k * tile_size_K + cur_tile_size_K]

                    if not spike_stored_in_buffer:
                        if buffer_state_act == "store single tile":
                            if is_conv:
                                stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K // (kernel_size * kernel_size) # on chip img2col
                            else:
                                stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K
                            stats.writes['g_act'] += cur_tile_size_M * cur_tile_size_K
                        elif buffer_state_act == "store row":
                            if is_conv:
                                stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K // (kernel_size * kernel_size)
                            else:
                                stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K
                            stats.writes['g_act'] += cur_tile_size_M * cur_tile_size_K
                        elif buffer_state_act == "store all" and n == 0:
                            if is_conv:
                                stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K // (kernel_size * kernel_size)
                            else:
                                stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K
                            stats.writes['g_act'] += cur_tile_size_M * cur_tile_size_K
                    if not weight_stored_in_buffer:
                        if buffer_state_wgt == "store single tile":
                            stats.reads['dram'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                            stats.writes['g_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                        elif buffer_state_wgt == "store col" and m == 0:
                            stats.reads['dram'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                            stats.writes['g_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                        elif buffer_state_wgt == "store all" and m == 0:
                            stats.reads['dram'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                            stats.writes['g_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits

                    if self.accelerator.product_sparsity:
                        preprocess_act, prefix_array = find_product_sparsity(cur_act)
                        prosparsity_cycles = get_prosparsity_cycles(cur_act) + cur_act.shape[0] // self.accelerator.num_popcnt
                        stats.reads['g_act'] += cur_tile_size_M * cur_tile_size_K
                        stats.reads['g_wgt'] += torch.sum(preprocess_act != 0).item() * cur_tile_size_N * operator.weight_tensor.nbits
                        # find the row in preprocess_act that is all zero, if all zero originally, no cycles needed, if not, need one cycle
                        nnz_each_row = torch.sum(preprocess_act != 0, dim=-1)
                        nnz_each_row_ori = torch.sum(cur_act != 0, dim=-1)
                        all_zero_row = nnz_each_row == 0
                        all_zero_row_ori = nnz_each_row_ori == 0
                        if self.accelerator.issue_type == 1:
                            forest = construct_prosparsity_forest(prefix_array)
                            depth = nx.dag_longest_path_length(forest)
                            cur_issue_cycles = depth // 4 * cur_tile_size_M
                        else:
                            cur_issue_cycles = 0
                        cur_compute_cycles = torch.sum(preprocess_act != 0).item() + torch.sum(all_zero_row).item() - torch.sum(all_zero_row_ori).item()
                        compute_cycles += max(cur_issue_cycles, cur_compute_cycles)


                        num_all_zero_row.append(torch.sum(all_zero_row_ori).item())
                        num_EM_row.append(torch.sum(all_zero_row).item() - torch.sum(all_zero_row_ori).item())
                        num_PM_row.append(torch.sum(prefix_array != -1).item() - torch.sum(all_zero_row).item() + torch.sum(all_zero_row_ori).item())
                        num_other_row.append(cur_tile_size_M - torch.sum(prefix_array != -1).item() - torch.sum(all_zero_row_ori).item())

                        preprocess_cycles += prosparsity_cycles
                        orginial_compute_cycles += torch.sum(cur_act != 0).item()
                        stats.num_ops += torch.sum(preprocess_act != 0).item() * cur_tile_size_N

                        rank_one_prefix.append(torch.sum(prefix_array != -1).item())
                    
                    elif not self.accelerator.dense:
                        compute_cycles += torch.sum(cur_act != 0).item()

                        stats.reads['g_act'] += cur_tile_size_M * cur_tile_size_K
                        stats.reads['g_wgt'] += torch.sum(cur_act != 0).item() * cur_tile_size_N * operator.weight_tensor.nbits

                        stats.num_ops += torch.sum(cur_act != 0).item() * cur_tile_size_N
                    
                    elif self.accelerator.dense:
                        compute_cycles += cur_tile_size_M * cur_tile_size_K

                        stats.reads['g_act'] += cur_tile_size_M * cur_tile_size_K
                        stats.reads['g_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits


                    # write results to partial sum buffer
                    stats.reads['g_psum'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits
                    stats.writes['g_psum'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits


                    if self.accelerator.product_sparsity and self.track_sparsity_increment:
                        ori_nnzs.append(torch.sum(cur_act != 0).item())
                        processed_nnzs.append(torch.sum(preprocess_act != 0).item())
                        ops_nnzs.append(torch.sum(preprocess_act != 0).item() + torch.sum(all_zero_row).item() - torch.sum(all_zero_row_ori).item())
                        total_elements.append(cur_tile_size_M * cur_tile_size_K)


                    if self.test_rank_two:
                        rank_two_act, num_prefix = find_rank_two_product_sparsity(cur_act)
                        rank_two_nnzs.append(torch.sum(rank_two_act != 0).item())

                        rank_two_prefix.append(num_prefix)

        init_mem_access = 0
        if not weight_stored_in_buffer:
            init_mem_access += min(tile_size_K, K) * min(tile_size_N, N) * operator.weight_tensor.nbits # read the first tile from dram to buffer
        if not spike_stored_in_buffer:
            init_mem_access += min(tile_size_K, K) * min(tile_size_M, M)

        total_mem_access = stats.reads['dram'] + stats.writes['dram']
        middle_mem_access = total_mem_access - init_mem_access
        init_latency = init_mem_access // self.accelerator.mem_if_width
        middle_latency = middle_mem_access // self.accelerator.mem_if_width
        stats.compute_cycles = max(compute_cycles, preprocess_cycles)
        stats.preprocess_stall_cycles = max(0, preprocess_cycles - compute_cycles)
        stats.mem_stall_cycles = init_latency + max(0, middle_latency - stats.compute_cycles)
        stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles


        print(operator.name)
        print("original compute cycles: ", orginial_compute_cycles)
        print("compute cycles: ", compute_cycles)
        print("preprocess cycles: ", preprocess_cycles)
        print("total cycles: ", stats.total_cycles)

        return stats
                    
    
    def run_conv2d(self, operator: Conv2D, spike_stored_in_buffer=False):
        eq_fc = conv2d_2_fc(operator)
        return self.run_fc(eq_fc, spike_stored_in_buffer)
    
    def run_conv2d_cuda(self, operator: Conv2D, spike_stored_in_buffer=False):
        eq_fc = conv2d_2_fc(operator)
        return self.run_fc_cuda(eq_fc, spike_stored_in_buffer)
    
    def run_maxpool(self, operator: MaxPool2D):
        # we ignore maxpool since it can be done light-weight when storing spike back to memory
        stats = Stats()
        return stats
    
    def run_LIF(self, operator: LIFNeuron, last_tile_size_M=256, last_tile_size_N=128):
        stats = Stats()
        stats.reads['g_psum'] = operator.activation_tensor.get_size()
        spike_stored_in_buffer = False
        num_round = ceil_a_by_b(operator.num_neuron * operator.batch_size, self.accelerator.LIF_array_size)
        compute_cycles = num_round * operator.time_steps * 2 # one cycle for addition one cycle for mutiplication
        tile_size_M = last_tile_size_M
        tile_size_N = last_tile_size_N
        num_neuron_in_tile = tile_size_M * tile_size_N // operator.time_steps
        latency = ceil_a_by_b(num_neuron_in_tile, self.accelerator.LIF_array_size) * operator.time_steps * 2
        latency = min(latency, compute_cycles)
        stats.compute_cycles = compute_cycles
        stats.LIF_latency = latency
        if operator.output_tensor.get_size() < self.accelerator.sram_size['act']:
            stats.writes['g_act'] = operator.output_tensor.get_size()
            spike_stored_in_buffer = True
        else:
            stats.writes['dram'] = operator.output_tensor.get_size()
            mem_cycles = operator.output_tensor.get_size() // self.accelerator.mem_if_width
            stats.mem_stall_cycles += max(0, mem_cycles - stats.compute_cycles)
            # no extra memory latency since spike is generated in streaming manner, each time 32 bits
        stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles

        print(operator.name)
        print("compute cycles: ", compute_cycles)
        print("latency: ", latency)
        print("total cycles: ", stats.total_cycles)
        return stats, spike_stored_in_buffer
    
    def run_attention(self, operator: Attention, spike_stored_in_buffer=False):
        stats = Stats()

        out_spike_stored_in_buffer = False

        if operator.attention_type == 'spikformer' or operator.attention_type == 'spikebert':
            # change the q k v multiplication order according to matrix shape, reduce computation overhead
            eq_sequence_length = max(operator.sequence_length, operator.dim_per_head)
            eq_dim_per_head = min(operator.sequence_length, operator.dim_per_head)

            # compute k*v first
            if operator.sequence_length > operator.dim_per_head:
                act_k = operator.act_k_tensor.sparse_map
                act_v = operator.act_v_tensor.sparse_map
            else:
                act_k = operator.act_k_tensor.sparse_map
                act_v = operator.act_q_tensor.sparse_map

            # then compute q * kv
            if operator.sequence_length > operator.dim_per_head:
                act_q = operator.act_q_tensor.sparse_map
            else:
                act_q = operator.act_v_tensor.sparse_map
            act_q = act_q.reshape([operator.time_steps, operator.batch_size, eq_sequence_length, operator.num_head, eq_dim_per_head]).permute(0, 1, 3, 2, 4).contiguous()
            act_k = act_k.reshape([operator.time_steps, operator.batch_size, eq_sequence_length, operator.num_head, eq_dim_per_head]).permute(0, 1, 3, 2, 4).contiguous()
            act_v = act_v.reshape([operator.time_steps, operator.batch_size, eq_sequence_length, operator.num_head, eq_dim_per_head]).permute(0, 1, 3, 2, 4).contiguous()
            for t in range(operator.time_steps):
                for b in range(operator.batch_size):
                    for h in range(operator.num_head):
                        cur_act_k = act_k[t, b, h, :, :].transpose(0, 1).contiguous()
                        name = f"{operator.name}_t{t}_b{b}_h{h}_kv"
                        input_dim = eq_sequence_length
                        output_dim = eq_dim_per_head
                        sequence_length = eq_dim_per_head
                        time_steps = 1 # no reuse between time steps
                        batch_size = 1
                        eq_fc_1 = FC(name, input_dim, output_dim, sequence_length, batch_size, time_steps)
                        eq_fc_1.activation_tensor.sparse_map = cur_act_k
                        out_weight_stored_in_buffer = eq_fc_1.output_tensor.get_size() < self.accelerator.sram_size['wgt']
                        if not out_weight_stored_in_buffer:
                            stats.writes['dram'] += eq_fc_1.weight_tensor.get_size() // 8
                        cur_stats_kv = self.run_fc(eq_fc_1, spike_stored_in_buffer, spike_stored_in_buffer)
                        stats.total_cycles += cur_stats_kv.total_cycles
                        stats.mem_stall_cycles += cur_stats_kv.mem_stall_cycles
                        stats.compute_cycles += cur_stats_kv.compute_cycles
                        stats += cur_stats_kv

                        cur_act_q = act_q[t, b, h, :, :]
                        name = f"{operator.name}_t{t}_b{b}_h{h}_qkv"
                        input_dim = eq_dim_per_head
                        output_dim = eq_dim_per_head
                        sequence_length = eq_sequence_length
                        time_steps = 1
                        batch_size = 1
                        eq_fc_2 = FC(name, input_dim, output_dim, sequence_length, batch_size, time_steps)
                        eq_fc_2.activation_tensor.sparse_map = cur_act_q
                        cur_stats_qkv = self.run_fc(eq_fc_2, spike_stored_in_buffer, out_weight_stored_in_buffer)
                        stats.total_cycles += cur_stats_qkv.total_cycles
                        stats.mem_stall_cycles += cur_stats_qkv.mem_stall_cycles
                        stats.compute_cycles += cur_stats_qkv.compute_cycles
                        stats += cur_stats_qkv

        elif operator.attention_type == 'sdt':
            init_mem_access = 0
            if not spike_stored_in_buffer:
                init_mem_access += operator.sequence_length * operator.dim_per_head * 3
                stats.reads['dram'] += operator.act_q_tensor.sparse_map.numel() * 3


            num_op = operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head * operator.sequence_length

            stats.compute_cycles += num_op // self.accelerator.adder_array_size
            stats.reads['g_act'] += operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head * operator.sequence_length * 2
            stats.writes['g_psum'] += operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head
            lif = LIFNeuron('lif_attn', operator.dim_per_head * operator.num_head, operator.batch_size, operator.time_steps)
            lif_stats, _ = self.run_LIF(lif, self.accelerator.adder_array_size, 1)
            stats.compute_cycles += max(0, lif_stats.total_cycles - stats.compute_cycles)
            stats.reads['g_act'] += num_op * 2
            out_spike_stored_in_buffer = False
            if operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head * operator.sequence_length < self.accelerator.sram_size['act']:
                stats.writes['g_act'] += operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head * operator.sequence_length
                out_spike_stored_in_buffer = True
            else:
                stats.writes['dram'] += operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head * operator.sequence_length

            stats.compute_cycles += num_op // self.accelerator.adder_array_size
            
            total_mem_access = stats.reads['dram'] + stats.writes['dram']
            middle_mem_access = total_mem_access - init_mem_access
            init_latency = init_mem_access // self.accelerator.mem_if_width
            stats.mem_stall_cycles = init_latency + max(0, middle_mem_access // self.accelerator.mem_if_width - stats.compute_cycles)
            stats.total_cycles = stats.compute_cycles + stats.mem_stall_cycles

        elif operator.attention_type == 'spikingbert':
            init_mem_access = 0
            init_mem_access += operator.sequence_length * operator.dim_per_head * (8 + 1) # q is 8bits k, kv is 1 bit
            stats.reads['dram'] += operator.act_k_tensor.sparse_map.numel() * (8 + 1 + 1)
            stats.writes['g_act'] += operator.act_k_tensor.sparse_map.numel() + operator.act_v_tensor.sparse_map.numel()
            stats.writes['g_wgt'] += operator.act_k_tensor.sparse_map.numel() * 8

            act_k = operator.act_k_tensor.sparse_map
            act_v = operator.act_v_tensor.sparse_map
            act_k = act_k.reshape([operator.time_steps, operator.batch_size, operator.sequence_length, operator.num_head, operator.dim_per_head]).permute(0, 1, 3, 2, 4).contiguous()
            act_v = act_v.reshape([operator.time_steps, operator.batch_size, operator.sequence_length, operator.num_head, operator.dim_per_head]).permute(0, 1, 3, 2, 4).contiguous()
            for t in range(operator.time_steps):
                for b in range(operator.batch_size):
                    for h in range(operator.num_head):
                        cur_act_k = act_k[t, b, h, :, :]
                        name = f"{operator.name}_t{t}_b{b}_h{h}_qk"
                        input_dim = operator.dim_per_head
                        output_dim = operator.sequence_length
                        sequence_length = operator.sequence_length
                        time_steps = 1
                        batch_size = 1
                        eq_fc_qk = FC(name, input_dim, output_dim, sequence_length, batch_size, time_steps)
                        eq_fc_qk.activation_tensor.sparse_map = cur_act_k
                        
                        cur_stats_qk = self.run_fc(eq_fc_qk, True, True)
                        qk_cycles = cur_stats_qk.total_cycles

                        # softmax
                        SFU_cycles = operator.sequence_length * operator.sequence_length // self.accelerator.num_exp # exponentiate
                        adder_cycles = operator.sequence_length * operator.sequence_length // self.accelerator.adder_array_size # sum
                        SFU_cycles += operator.sequence_length // self.accelerator.num_div # reciprocal
                        multiplier_cycles = operator.sequence_length * operator.sequence_length // self.accelerator.multiplier_array_size # normalize

                        softmax_cycles = SFU_cycles + multiplier_cycles

                        cur_act_v = act_v[t, b, h, :, :].transpose(0, 1).contiguous()
                        name = f"{operator.name}_t{t}_b{b}_h{h}_sv"
                        input_dim = operator.sequence_length
                        output_dim = operator.sequence_length
                        sequence_length = operator.dim_per_head
                        time_steps = 1
                        batch_size = 1
                        eq_fc_sv = FC(name, input_dim, output_dim, sequence_length, batch_size, time_steps)
                        eq_fc_sv.activation_tensor.sparse_map = cur_act_v
                        cur_stats_sv = self.run_fc(eq_fc_sv, True, True)

                        sv_cycles = cur_stats_sv.total_cycles

                        stats.compute_cycles += max(qk_cycles + sv_cycles + adder_cycles, softmax_cycles)

            stats.total_cycles = stats.compute_cycles


        else:
            raise Exception("unsupported attention type")
        
        print(operator.name)
        print("total cycles: ", stats.total_cycles)
        print("compute cycles: ", stats.compute_cycles)
        if operator.attention_type == 'spikformer' or operator.attention_type == 'spikebert':
            out_spike_stored_in_buffer = False

        return stats, out_spike_stored_in_buffer
    
    def run_layernorm(self, operator: LayerNorm):
        stats = Stats()

        M,N = np.prod(operator.activation_tensor.shape[:-1]), operator.activation_tensor.shape[-1]
        tile_size_M = self.accelerator.SpMM_tile_size_M
        tile_num_M = ceil_a_by_b(M, tile_size_M)

        stats.reads['g_psun'] = operator.activation_tensor.get_size()
        stats.writes['dram'] = operator.activation_tensor.get_size() # store value back to dram first
        stats.reads['dram'] = operator.activation_tensor.get_size()
        stats.writes['g_wgt'] = operator.activation_tensor.get_size()

        stats.reads['g_act'] = operator.activation_tensor.get_size() # read out value
        stats.writes['g_act'] = operator.activation_tensor.get_size() # write back minus by mean value
        stats.reads['g_psum'] = operator.activation_tensor.get_size() # read out value
        stats.writes['g_psum'] = operator.activation_tensor.get_size() # write back squared value
        adder_cycles = 0
        multiplier_cycles = 0
        SFU_cycles = 0
        for m in range(tile_num_M):
            cur_tile_size_M = min(tile_size_M, M - m * tile_size_M)
            adder_cycles += cur_tile_size_M * N // self.accelerator.adder_array_size # get sum
            adder_cycles += cur_tile_size_M * N // self.accelerator.adder_array_size # minus by mean
            adder_cycles += cur_tile_size_M * N // self.accelerator.adder_array_size # get variance

        # only consider cycles for last tile since the computation is overlapped
        SFU_cycles += tile_size_M // self.accelerator.num_div # divide to get mean
        multiplier_cycles += tile_size_M * N // self.accelerator.multiplier_array_size # square
        SFU_cycles += tile_size_M // self.accelerator.num_div # get reciprocal of variance

        stats.compute_cycles = adder_cycles + multiplier_cycles + SFU_cycles
        stats.total_cycles = stats.compute_cycles

        print(operator.name)
        print("total cycles: ", stats.total_cycles)

        return stats

                            
    

def verify_product_sparsity(act_pro: torch.Tensor, act_bit: torch.Tensor, tree: nx.DiGraph):
    for i in range(act_pro.shape[0]):
        if tree.in_degree(i) == 0:
            continue
        prefix = list(tree.predecessors(i))
        prefix_row = act_bit[prefix[0]]
        overlap = torch.logical_and(prefix_row, act_pro[i])
        assert torch.sum(overlap) == 0
        act_pro[i] = torch.logical_or(act_pro[i], prefix_row)
    # check if two tensor are same
    assert torch.all(act_pro == act_bit)
    return True
    
def construct_prosparsity_forest(prefix_array: torch.Tensor):
    forest = nx.DiGraph()
    for i in range(prefix_array.shape[0]):
        if prefix_array[i].item() == -1:
            continue
        forest.add_edge(prefix_array[i].item(), i)
    return forest

def find_rank_two_product_sparsity(act: torch.Tensor):
    preprocessed_act = act.clone()
    num_prefix = 0
    for i in range(act.shape[0]):
        cur_row = act[i]
        nnz = torch.sum(cur_row != 0).item()
        if nnz < 2:
            continue
        and_result = torch.logical_and(cur_row, act)
        equalities = torch.eq(and_result, act)
        is_subset = torch.all(equalities, dim=-1)
        equalities = torch.eq(cur_row, act)
        is_equal = torch.all(equalities, dim=-1)
        is_bigger_index = torch.arange(act.shape[0]) >= i
        is_excluded = torch.logical_and(is_equal, is_bigger_index)
        is_real_subset = torch.logical_and(is_subset, ~is_excluded)
        if torch.sum(is_real_subset) == 0:
            continue
        
        subset_row = act[is_real_subset]
        subset_row_nnz = torch.sum(subset_row != 0, dim=-1)
        max_subset_size = torch.max(subset_row_nnz).item()
        prefix_row = subset_row[torch.argmax(subset_row_nnz)]
        prefix_row_nnz = torch.sum(prefix_row).item()
        rank_one_prefix_nnz = prefix_row_nnz
        if prefix_row_nnz > 0:
            num_prefix += 1
        if max_subset_size == torch.sum(cur_row).item():
            preprocessed_act[i] = torch.logical_xor(preprocessed_act[i], prefix_row)
            continue
        for j in range(subset_row.shape[0]):
            cur_subset_row = subset_row[j]
            if prefix_row_nnz == nnz:
                break
            if torch.sum(cur_subset_row) == 0:
                continue
            and_result = torch.logical_and(cur_subset_row, subset_row)
            and_sum = torch.sum(and_result, dim=-1)
            for k in range(and_sum.shape[0]):
                if and_sum[k].item() == 0:
                    if torch.sum(cur_subset_row) + torch.sum(subset_row[k]) > prefix_row_nnz:
                        prefix_row = torch.logical_or(cur_subset_row, subset_row[k])
                        prefix_row_nnz = torch.sum(prefix_row).item()
        if prefix_row_nnz > rank_one_prefix_nnz:
            num_prefix += 1
        preprocessed_act[i] = torch.logical_xor(preprocessed_act[i], prefix_row)
    return preprocessed_act, num_prefix

def get_prosparsity_cycles(act: torch.Tensor):

    nnz_each_row = torch.sum(act != 0, dim=-1)
    num_searched_row = torch.sum(nnz_each_row > 1).item()

    return num_searched_row

def find_product_sparsity(act: torch.Tensor):
    preprocessed_act = act.clone()
    prefix_array = torch.ones(act.shape[0])
    prefix_array = -prefix_array # set all to -1
    for i in range(act.shape[0]):
        cur_row = act[i]
        nnz = torch.sum(cur_row != 0).item()
        if nnz < 2:
            continue
        and_result = torch.logical_and(cur_row, act)
        equalities = torch.eq(and_result, act)
        is_subset = torch.all(equalities, dim=-1)
        equalities = torch.eq(cur_row, act)
        is_equal = torch.all(equalities, dim=-1)
        is_bigger_index = torch.arange(act.shape[0]) >= i
        is_excluded = torch.logical_and(is_equal, is_bigger_index)
        is_real_subset = torch.logical_and(is_subset, ~is_excluded)
        if torch.sum(is_real_subset) == 0:
            # CAM no match
            continue
        subset_row = act[is_real_subset]
        subset_row_nnz = torch.sum(subset_row != 0, dim=-1)
        max_subset_size = torch.max(subset_row_nnz).item()
        max_subset = subset_row[torch.argmax(subset_row_nnz)]
        subset_index = torch.nonzero(is_real_subset).flatten()
        subset_size = torch.sum(act[is_real_subset], dim=-1)
        max_subset_size, max_subset_index = torch.max(subset_size, dim=-1)
        if max_subset_size.item() < 1:
            continue
        prefix_array[i] = subset_index[max_subset_index].item()
        # if max_subset_size > 1: # can also reuse even when the size is 1
        preprocessed_act[i] = torch.logical_xor(preprocessed_act[i], max_subset)
    
    return preprocessed_act, prefix_array


                    
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description='Simulator')
    parser.add_argument('--type', type=str, default='Prosperity', help='type of accelerator')
    parser.add_argument('--adder_array_size', type=int, default=128, help='size of adder array')
    parser.add_argument('--LIF_array_size', type=int, default=32, help='size of LIF array')
    parser.add_argument('--tile_size_M', type=int, default=256, help='tile size M')
    parser.add_argument('--tile_size_K', type=int, default=16, help='tile size K')
    parser.add_argument('--bit_sparsity', action='store_true', default=False, help='bit sparsity mode, no product sparsity')
    parser.add_argument('--output_dir', type=str, default='../output', help='output directory')
    parser.add_argument('--dense', action='store_true', default=False, help='dense')
    parser.add_argument('--use_cuda', action='store_true', default=False, help='use cuda')
    parser.add_argument('--dse_mode', action='store_true', default=False, help='design space exploration mode')
    parser.add_argument('--sparse_analysis_mode', action='store_true', default=False, help='analyze the ProSparity and BitSparsity in extra models')
    parser.add_argument('--issue_type', type=int, default=2, help='Prosperity issue type (1 or 2)')

    args = parser.parse_args()


    # Adder 8 bit * 128

    accelerator = Accelerator(type=args.type, 
                              adder_array_size=args.adder_array_size, 
                              LIF_array_size=args.LIF_array_size, 
                              tile_size_M=args.tile_size_M, 
                              tile_size_K=args.tile_size_K,
                              product_sparsity=not args.bit_sparsity,
                              issue_type=args.issue_type,
                              dense=args.dense,
                              )
    
    ST_model_list = [
                     'spikformer_cifar10', 'spikformer_cifar10dvs', 'spikformer_cifar100', 
                     'sdt_cifar10', 'sdt_cifar10dvs', 'sdt_cifar100',
                     'spikebert_sst2', 'spikebert_mr', 'spikebert_sst5', 
                     'spikingbert_sst2', 'spikingbert_qqp', 'spikingbert_mnli',
                     ]
    
    SCNN_model_list = ['vgg16_cifar10', 'vgg16_cifar100', 
                       'resnet18_cifar10', 'resnet18_cifar100', 
                       ]
    stats_list = []

    run_ST = True
    run_SCNN = True
    run_single_model = False
    model_list = [] # test set

    if run_SCNN:
        model_list.extend(SCNN_model_list)
    if run_ST:
        model_list.extend(ST_model_list)
    if run_single_model:
        model_list = ['spikformer_cifar100',]

    if args.sparse_analysis_mode:
        model_list = ['vgg16_cifar10', 'vgg16_cifar100', 
                      'vgg9_cifar10', 'vgg9_cifar10dvs',
                       'resnet18_cifar10', 'resnet18_cifar100',
                       'lenet5_mnist',
                       'spikformer_cifar10', 'spikformer_cifar10dvs', 'spikformer_cifar100', 
                     'sdt_cifar10', 'sdt_cifar10dvs', 'sdt_cifar100',
                     'spikebert_sst2', 'spikebert_mr', 'spikebert_sst5', 
                     'spikingbert_sst2', 'spikingbert_qqp', 'spikingbert_mnli',
                        ]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dse_mode = args.dse_mode

    if not dse_mode and not args.sparse_analysis_mode:
        time_file = os.path.join(args.output_dir, "time.csv")
        energy_file = os.path.join(args.output_dir, "energy.csv")
        density_file = os.path.join(args.output_dir, "density.csv")

        if not os.path.exists(time_file):
            write_title(file_name=time_file, title=['model_name'] + model_list)
        if not os.path.exists(energy_file):
            write_title(file_name=energy_file, title=['model_name'] + model_list)
        if not os.path.exists(density_file):
            write_title(file_name=density_file, title=['model_name'] + model_list)

    elif dse_mode:
        if args.bit_sparsity:
            time_file = os.path.join(args.output_dir, "time_bit_sparsity.csv".format(args.tile_size_M, args.tile_size_K))
            density_file = os.path.join(args.output_dir, "density_bit_sparsity.csv".format(args.tile_size_M, args.tile_size_K))
            energy_file = None
        else:
            time_file = os.path.join(args.output_dir, "time_M{}_K{}.csv".format(args.tile_size_M, args.tile_size_K))
            density_file = os.path.join(args.output_dir, "density_M{}_K{}.csv".format(args.tile_size_M, args.tile_size_K))
            energy_file = None

        if not os.path.exists(time_file):
            write_title(file_name=time_file, title=['model_name'] + model_list)
        if not os.path.exists(density_file):
            write_title(file_name=density_file, title=['model_name'] + model_list)
    
    elif args.sparse_analysis_mode:
        density_file = os.path.join(args.output_dir, "density_analysis.csv")
        time_file = None
        energy_file = None

        if not os.path.exists(density_file):
            write_title(file_name=density_file, title=['model_name'] + model_list)

    for name in model_list:
        clear_global_stats()
        model_name = name.split('_')[0]
        spike_info_path = "../data/" + name + ".pkl"
        nn = create_network(model_name, spike_info_path)

        if args.type == 'LoAS':
            print(name)
        sim = Simulator(accelerator=accelerator, network=nn, benchmark_name=name, use_cuda=args.use_cuda)
        stats = sim.sim()
        stats_list.append(stats)

        if args.type == 'LoAS':
            continue
        if not dse_mode and not args.sparse_analysis_mode:
            energy = get_total_energy(stats, args.type.split('_')[0], name)
            print(f"total energy: {energy}")
        else:
            energy = None
        runtime = stats.total_cycles / (500 * 1000 * 1000) if stats.total_cycles is not None else None

        if time_file is not None:
            write_position(file_name=time_file, 
                        column_name=name, 
                        row_name=args.type,
                        data=runtime)
        if energy_file is not None:
            write_position(file_name=energy_file, 
                        column_name=name, 
                        row_name=args.type,
                        data=energy)
        filename_element_list = [args.type, 
                                 'ST' if not run_single_model and run_ST else None, 
                                 'SCNN' if not run_single_model and run_SCNN else None, 
                                 args.tile_size_M if args.type == 'Prosperity' else None, 
                                 args.tile_size_K if args.type == 'Prosperity' else None, 
                                 model_list[0] if run_single_model else None,
                                 "cuda" if args.use_cuda else "cpu",]
        stats_filename = '_'.join([str(e) for e in filename_element_list if e is not None])
        stats_file_path = os.path.join(args.output_dir, stats_filename + '.txt')


        with open(stats_file_path, 'a') as f:  # Open the file in append mode
            f.write(f"model: {name}\n")
            f.write(f"end to end time: {runtime}\n")
            f.write(f"total energy: {energy}\n")
            if args.type == 'Prosperity':
                f.write(f"mem access: {stats.reads['dram'] + stats.writes['dram']}\n")
                f.write(f"total cycles: {stats.total_cycles}\n")
                f.write(f"preprocess stall cycle: {stats.preprocess_stall_cycles}\n")
                f.write(f"bit density: {stats.bit_density}\n")
                f.write(f"product density: {stats.product_density}\n")
                f.write(f"mem stall cycle: {stats.mem_stall_cycles}\n")

                if density_file is not None:
                    write_position(file_name=density_file, 
                                   column_name=name, 
                                   row_name="bit density",
                                   data=stats.bit_density)
                    write_position(file_name=density_file,
                                     column_name=name, 
                                     row_name="product density",
                                     data=stats.product_density)

            f.write("\n")

