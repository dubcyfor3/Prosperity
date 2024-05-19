from networks import FC, Conv2D, MaxPool2D, LIFNeuron, Attention, create_network
from utils import ceil_a_by_b, img2col, get_density
import torch
import numpy as np
from collections import defaultdict


class Stats:
    def __init__(self):
        self.total_cycles = 0
        self.mem_stall_cycles = 0
        self.compute_cycles = 0
        self.num_ops = 0
        self.mem_namespace = ['dram', 'g_act', 'g_wgt', 'g_psum', 'l_act', 'l_wgt']
        self.reads = {space: 0 for space in self.mem_namespace}
        self.writes = {space: 0 for space in self.mem_namespace}


class Accelerator:
    def __init__(self, num_PE, sram_size, adder_width, mem_if_width=1024):
        self.num_PE = num_PE
        self.sram_size = {}
        self.sram_size['wgt'] = sram_size['wgt'] # global buffer
        self.sram_size['act'] = sram_size['act'] # global buffer
        self.sram_size['psum'] = sram_size['psum'] # global buffer
        self.sram_size['out'] = sram_size['out']
        self.adder_width = adder_width
        self.mem_if_width = mem_if_width

class Simulator:
    def __init__(self, accelerator: Accelerator, network: list):
        self.accelerator = accelerator
        self.network = network
    
    def run_simulation(self):
        stats = {}
        for operator in self.network:
            if isinstance(operator, FC):
                stats[operator.name] = self.run_fc(operator)
            elif isinstance(operator, Conv2D):
                stats[operator.name] = self.run_conv2d(operator)           
            else:
                pass
    
    def run_fc(self, operator: FC):
        stats_list = [Stats() for _ in range(self.accelerator.num_PE)]
        total_stats = Stats()
        assert operator.activation_tensor.shape[-1] == operator.weight_tensor.shape[0]
        # reshape activation tensor
        input_shape = operator.activation_tensor.shape
        input_shape = [np.prod(input_shape[:-1]), input_shape[-1]]
        input_tensor = operator.activation_tensor.sparse_map.reshape(input_shape)

        M, K, N = input_shape[0], input_shape[1], operator.weight_tensor.shape[1]
        tile_size_M = 64
        tile_size_K = 128 # dense format 128 bit, sparse format 7 bit * num_nonzero
        tile_size_N = 8 # 8 bits per element, 64 bit in a line
        tile_num_M = ceil_a_by_b(M, tile_size_M)
        tile_num_K = ceil_a_by_b(K, tile_size_K)
        tile_num_N = ceil_a_by_b(N, tile_size_N)
        cur_M_idx = 0

        # store as CSR format
        input_size = torch.sum(input_tensor != 0).item() * int(np.log2(input_shape[1])) + input_shape[0] * int(np.log2(input_shape[0]))
        input_tile_size_avg = input_size // (tile_num_M * tile_num_K)
        weight_tile_size = tile_size_K * tile_size_N * operator.weight_tensor.nbits

        buffer_state_act = None
        if input_tile_size_avg * (tile_num_M * tile_num_K) < self.accelerator.sram_size['act']:
            buffer_state_act = "store all"
        elif input_tile_size_avg * tile_num_K < self.accelerator.sram_size['act']:
            buffer_state_act = "store row"
        elif input_tile_size_avg < self.accelerator.sram_size['act']:
            buffer_state_act = "store single tile"
        else:
            raise Exception("single tile cannot fit in sram act buffer")
        
        buffer_state_wgt = None
        if weight_tile_size * (tile_num_K * tile_num_N) < self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store all"
        elif weight_tile_size * tile_num_K < self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store col"
        elif weight_tile_size < self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store single tile"
        else:
            raise Exception("single tile cannot fit in sram wgt buffer")

        # init dram read that cannot overlap with compute
        init_dram_read_act = input_tile_size_avg * min(self.accelerator.num_PE, tile_num_K)
        init_dram_read_wgt = weight_tile_size * min(self.accelerator.num_PE, tile_num_K * tile_num_N)


        PE_idx = 0
        for m in range(tile_num_M):
            cur_tile_size_M = min(tile_size_M, M - cur_M_idx)
            cur_M_idx += cur_tile_size_M
            cur_N_idx = 0
            for n in range(tile_num_N):
                cur_tile_size_N = min(tile_size_N, N - cur_N_idx)
                cur_N_idx += cur_tile_size_N
                cur_K_idx = 0
                for k in range(tile_num_K):
                    cur_tile_size_K = min(tile_size_K, K - cur_K_idx)
                    cur_K_idx += cur_tile_size_K

                    stats = stats_list[PE_idx]
                    PE_idx = (PE_idx + 1) % self.accelerator.num_PE

                    cur_act = input_tensor[cur_M_idx - cur_tile_size_M:cur_M_idx, cur_K_idx - cur_tile_size_K:cur_K_idx]
                    cur_nnz = torch.sum(cur_act != 0).item()
                    if buffer_state_act == "store single tile":
                        stats.reads['dram'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))
                        stats.writes['g_act'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))
                    elif buffer_state_act == "store row" and n == 0:
                        stats.reads['dram'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))
                        stats.writes['g_act'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))
                    elif buffer_state_act == "store all" and n == 0:
                        stats.reads['dram'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))
                        stats.writes['g_act'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))
                    
                    if buffer_state_wgt == "store single tile":
                        stats.reads['dram'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                        stats.writes['g_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                    elif buffer_state_wgt == "store col":
                        stats.reads['dram'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                        stats.writes['g_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                    elif buffer_state_wgt == "store all" and m == 0:
                        stats.reads['dram'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                        stats.writes['g_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                    
                    stats.reads['g_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                    stats.writes['l_wgt'] += cur_tile_size_K * cur_tile_size_N * operator.weight_tensor.nbits
                    # read activation in sparse format
                    stats.reads['g_act'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))
                    stats.writes['l_act'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))
                    stats.reads['l_act'] += cur_nnz * int(np.log2(cur_tile_size_K)) + cur_tile_size_M * int(np.log2(cur_tile_size_M))

                    # assume idx % self.accelerator.adder_width == 0 has same bank
                    # assert cur_tile_size_K % self.accelerator.adder_width == 0
                    if cur_tile_size_K % self.accelerator.adder_width != 0:
                        # pad zeros to make sure cur_tile_size_K % self.accelerator.adder_width == 0
                        pad_size = self.accelerator.adder_width - cur_tile_size_K % self.accelerator.adder_width
                        cur_act = torch.cat([cur_act, torch.zeros(cur_tile_size_M, pad_size)], dim=-1)
                    cur_act = cur_act.reshape((cur_tile_size_M, cur_tile_size_K // self.accelerator.adder_width, self.accelerator.adder_width))
                    num_spike_per_bank = torch.sum(cur_act, dim=-2)
                    max_spike_per_bank = torch.max(num_spike_per_bank, dim=-1)[0]
                    adder_tree_cycles = torch.sum(max_spike_per_bank) + int(np.log2(self.accelerator.adder_width)) # assume adder tree is pipelined

                    stats.reads['l_wgt'] += cur_nnz * cur_tile_size_N * operator.weight_tensor.nbits
                    stats.writes['g_psum'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits

                    if k == tile_num_K - 1:
                        stats.reads['g_psum'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits
                        stats.writes['dram'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits

                    stats.compute_cycles += adder_tree_cycles.item()

        init_dram_access = init_dram_read_act + init_dram_read_wgt
        total_dram_access = sum([stats.reads['dram'] for stats in stats_list]) + sum([stats.writes['dram'] for stats in stats_list])
        middle_dram_access = total_dram_access - init_dram_access
        init_mem_cycles = (init_dram_access) // self.accelerator.mem_if_width
        middle_mem_cycles = (middle_dram_access) // self.accelerator.mem_if_width
        total_stats.compute_cycles = max([stats.compute_cycles for stats in stats_list])
        total_stats.mem_stall_cycles = init_mem_cycles + max(0, middle_mem_cycles - total_stats.compute_cycles)
        total_stats.total_cycles = total_stats.compute_cycles + total_stats.mem_stall_cycles
        

        return stats_list, total_stats
    def conv2d_2_fc(self, operator: Conv2D) -> FC:
        eq_input_dim  = operator.kernel_size * operator.kernel_size * operator.input_channel
        eq_sequence_length = operator.output_H * operator.output_H
        eq_output_dim = operator.output_channel
        eq_sparse_map = img2col(operator.activation_tensor.sparse_map, operator.kernel_size, operator.stride, operator.padding)
        eq_fc = FC(operator.name + '_2fc', eq_input_dim, eq_output_dim, eq_sequence_length, operator.batch_size, operator.time_steps)
        eq_fc.activation_tensor.sparse_map = eq_sparse_map
        return eq_fc
    
    def run_conv2d(self, operator: Conv2D):
        # eq_input_dim  = operator.kernel_size * operator.kernel_size * operator.input_channel
        # eq_sequence_length = operator.output_H * operator.output_H
        # eq_output_dim = operator.output_channel
        # eq_sparse_map = img2col(operator.activation_tensor.sparse_map, operator.kernel_size, operator.stride, operator.padding)
        # eq_fc = FC(operator.name + '_2fc', eq_input_dim, eq_output_dim, eq_sequence_length, operator.batch_size, operator.time_steps)
        # eq_fc.activation_tensor.sparse_map = eq_sparse_map
        eq_fc = self.conv2d_2_fc(operator)
        return self.run_fc(eq_fc)
    
    def find_largest_subset(self, operator: FC):
        input_shape = operator.activation_tensor.shape
        input_shape = [np.prod(input_shape[:-1]), input_shape[-1]]
        input_tensor = operator.activation_tensor.sparse_map.reshape(input_shape)
        tile_size_M = 256
        tile_size_K = 16
        nnz_list = []
        total_nnzs = torch.sum(input_tensor != 0).item()
        reduced_nnz = total_nnzs
        for m in range(0, input_shape[0], tile_size_M):
            for k in range(0, input_shape[1], tile_size_K):
                cur_tile_size_M = min(tile_size_M, input_shape[0] - m)
                cur_tile_size_K = min(tile_size_K, input_shape[1] - k)
                subset_dict = {}
                cur_tensor = input_tensor[m:m+cur_tile_size_M, k:k+cur_tile_size_K]
                for i in range(cur_tile_size_M):
                    cur_row = cur_tensor[i]
                    nnz = torch.sum(cur_row != 0).item()
                    # nnz_list.append(nnz)
                    if nnz < 2:
                        continue
                    for j in range(cur_tile_size_M):
                        if i == j or torch.sum(cur_tensor[j] != 0).item() < 2 or (torch.equal(cur_tensor[j], cur_row) and i < j):
                            continue
                        # if (torch.sum(cur_tensor[j] != 0).item() < 2):
                        #     continue
                        and_result = torch.logical_and(cur_row, cur_tensor[j])
                        is_subset = torch.equal(and_result, cur_tensor[j]) # j is subset of i
                        # if torch.equal(cur_tensor[j], cur_row) and i < j:
                        #     continue
                        # check if i exist in subset_dict
                        if is_subset:
                            j_nnz = torch.sum(cur_tensor[j] != 0).item()
                            if i in subset_dict:
                                prev_j, prev_nnz = subset_dict[i]
                                if j_nnz > prev_nnz:
                                    subset_dict[i] = (j, j_nnz)
                            else:
                                subset_dict[i] = (j, j_nnz)
                for key, value in subset_dict.items():
                    reduced_nnz -= value[1] - 1
        print("preprocessed nnz: ", reduced_nnz)
        print("total nnz: ", total_nnzs)
        print("percentage: ", reduced_nnz / total_nnzs)
        return reduced_nnz, total_nnzs

                    
if __name__ == '__main__':
    accelerator = Accelerator(num_PE=8, sram_size={'wgt': 64000000, 'act': 64000000, 'psum': 64, 'out': 64}, adder_width=8)
    nn = create_network('spikformer', 'test.pkl')
    sim = Simulator(accelerator, nn)
    # sim.run_simulation()
    # operator = nn[10]
    # sim.find_common_sequence(operator)
    # nn = nn[10:]
    pre_nnzs = []
    total_nnzs = []
    for operator in nn:
        if isinstance(operator, Conv2D):
            eq_fc = sim.conv2d_2_fc(operator)
            print(eq_fc.name)
            pre_nnz, total_nnz = sim.find_largest_subset(eq_fc)
        if isinstance(operator, FC):
            print(operator.name)
            pre_nnz, total_nnz = sim.find_largest_subset(operator)

        pre_nnzs.append(pre_nnz)
        total_nnzs.append(total_nnz)
    
    print("preprocessed nnzs: ", sum(pre_nnzs))
    print("total nnzs: ", sum(total_nnzs))
    print("percentage: ", sum(pre_nnzs) / sum(total_nnzs))

    # fc = nn[10]
    # # create a torch tensor, that is a upper triangular matrix
    # b = torch.rand(100, 100)
    # # b = torch.eye(100)
    # b = b.to(torch.bool)

    # fc.activation_tensor.sparse_map = b
    # fc.activation_tensor.shape = b.shape
    # sim.find_largest_subset(fc)


