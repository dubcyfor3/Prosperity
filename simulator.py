from networks import FC, Conv2D, MaxPool2D, LIFNeuron, Attention, create_network, conv2d_2_fc
from utils import ceil_a_by_b, img2col, get_density
import torch
import numpy as np
from collections import defaultdict
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO)

class Stats:
    def __init__(self):
        self.total_cycles = 0
        self.mem_stall_cycles = 0
        self.compute_cycles = 0
        self.num_ops = 0
        self.LIF_latency = 0
        self.mem_namespace = ['dram', 'g_act', 'g_wgt', 'g_psum', 'l_act', 'l_wgt']
        self.reads = {space: 0 for space in self.mem_namespace}
        self.writes = {space: 0 for space in self.mem_namespace}

    def __add__(self, other):
        if not isinstance(other, Stats):
            raise Exception("unsupported type")
        else:
            added_stats = Stats()
            added_stats.total_cycles = self.total_cycles # handle cycles manually
            added_stats.mem_stall_cycles = self.mem_stall_cycles
            added_stats.compute_cycles = self.compute_cycles
            added_stats.num_ops = self.num_ops + other.num_ops
            added_stats.LIF_latency = 0
            added_stats.mem_namespace = self.mem_namespace
            added_stats.reads = {space: self.reads[space] + other.reads[space] for space in self.mem_namespace}
            added_stats.writes = {space: self.writes[space] + other.writes[space] for space in self.mem_namespace}
            
            return added_stats


class Accelerator:
    def __init__(self, num_PE, sram_size, adder_array_size=128, LIF_array_size=32, mem_if_width=1024):
        self.num_popcnt = num_PE
        self.sram_size = {}
        self.sram_size['wgt'] = sram_size['wgt'] # global buffer
        self.sram_size['act'] = sram_size['act'] # global buffer
        self.sram_size['psum'] = sram_size['psum'] # global buffer
        self.adder_array_size = adder_array_size
        self.LIF_array_size = LIF_array_size
        self.bit_operation_width = 32
        self.bo_array_size = 4
        self.mem_if_width = mem_if_width

class Simulator:
    def __init__(self, accelerator: Accelerator, network: list):
        self.accelerator = accelerator
        self.network = network
    
    def run_simulation(self):
        stats = OrderedDict()
        spike_stored_in_buffer = False
        for operator in self.network:
            if isinstance(operator, FC):
                stats[operator.name] = self.run_fc(operator, spike_stored_in_buffer)
            elif isinstance(operator, Conv2D):
                stats[operator.name] = self.run_conv2d(operator, spike_stored_in_buffer)
            elif isinstance(operator, LIFNeuron):
                stats[operator.name], spike_stored_in_buffer = self.run_LIF(operator)
            elif isinstance(operator, Attention):
                stats[operator.name], spike_stored_in_buffer = self.run_attention(operator, spike_stored_in_buffer)
            elif isinstance(operator, MaxPool2D):
                stats[operator.name] = self.run_maxpool(operator)
            else:
                raise Exception("unsupported operator")

        total_stats = Stats()
        last_stats_key, last_stats = next(iter(stats.items()))
        for key, value in stats.items():
            total_stats += value
            # lif can be overlapped with fc and conv
            if key.startswith('lif') and (last_stats_key.startswith('fc') or last_stats_key.startswith('conv') or last_stats_key.startswith('attention')):
                last_stats_cycles = last_stats.total_cycles
                if last_stats_cycles + value.LIF_latency >= value.total_cycles:
                    total_stats.total_cycles += value.LIF_latency
                else:
                    total_stats.total_cycles += value.total_cycles - last_stats_cycles
            else:
                total_stats.total_cycles += value.total_cycles
            last_stats_key, last_stats = key, value

        
        print("total cycles: ", total_stats.total_cycles)
        print("time", total_stats.total_cycles / (500 * 1024 * 1024))
        print("total ops: ", total_stats.num_ops)
        print("mem access", total_stats.reads['dram'] + total_stats.writes['dram'])
        print("buffer access", total_stats.reads['g_act'] + total_stats.writes['g_act'] + total_stats.reads['g_wgt'] + total_stats.writes['g_wgt'] + total_stats.reads['g_psum'] + total_stats.writes['g_psum'])

        cycles_conv = 0
        cycles_fc = 0
        cycles_attn = 0
        for key, value in stats.items():
            if key.startswith('fc'):
                cycles_fc += value.total_cycles
            if key.startswith('conv'):
                cycles_conv += value.total_cycles
            if key.startswith('attention'):
                cycles_attn += value.total_cycles

        print("conv percentage: ", cycles_conv / total_stats.total_cycles)
        print("fc percentage: ", cycles_fc / total_stats.total_cycles)
        print("attn percentage: ", cycles_attn / total_stats.total_cycles)
    
    def run_fc(self, operator: FC, spike_stored_in_buffer=False, weight_stored_in_buffer=False):
        stats = Stats()
        assert operator.activation_tensor.shape[-1] == operator.weight_tensor.shape[0]
        # reshape activation tensor
        input_shape = operator.activation_tensor.shape
        input_shape = [np.prod(input_shape[:-1]), input_shape[-1]]
        input_tensor = operator.activation_tensor.sparse_map.reshape(input_shape)
        M, K, N = input_shape[0], input_shape[1], operator.weight_tensor.shape[1]
        tile_size_M = 256
        tile_size_K = 16
        tile_size_N = self.accelerator.adder_array_size
        tile_num_M = ceil_a_by_b(M, tile_size_M)
        tile_num_K = ceil_a_by_b(K, tile_size_K)
        tile_num_N = ceil_a_by_b(N, tile_size_N)

        act_tile_size = tile_size_M * tile_size_K * 1 # spike is one bit
        wgt_tile_size = tile_size_K * tile_size_N * operator.weight_tensor.nbits

        if M * K < self.accelerator.sram_size['act']:
            buffer_state_act = "store all"
        elif act_tile_size * tile_num_K < self.accelerator.sram_size['act']:
            buffer_state_act = "store row"
        elif act_tile_size < self.accelerator.sram_size['act']:
            buffer_state_act = "store single tile"
        else:
            raise Exception("single tile cannot fit in sram act buffer")
        
        if K * N * operator.weight_tensor.nbits < self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store all"
        elif wgt_tile_size * tile_num_K < self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store col"
        elif wgt_tile_size < self.accelerator.sram_size['wgt']:
            buffer_state_wgt = "store single tile"

        # 1. load activation and weight to sram
        # 2. load activation and weight to local buffer
        # 3. do preprocess to the activation
        # 4. start computation
        compute_cycles = 0
        preprocess_cycles = 0
        orginial_compute_cycles = 0


        for m in range(tile_num_M):
            for n in range(tile_num_N):
                for k in range(tile_num_K):
                    cur_tile_size_M = min(tile_size_M, M - m * tile_size_M)
                    cur_tile_size_K = min(tile_size_K, K - k * tile_size_K)
                    cur_tile_size_N = min(tile_size_N, N - n * tile_size_N)
                    cur_act = input_tensor[m * tile_size_M: m * tile_size_M + cur_tile_size_M, k * tile_size_K: k * tile_size_K + cur_tile_size_K]

                    if not spike_stored_in_buffer:
                        if buffer_state_act == "store single tile":
                            stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K
                            stats.writes['g_act'] += cur_tile_size_M * cur_tile_size_K
                        elif buffer_state_act == "store row" and n == 0:
                            stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K
                            stats.writes['g_act'] += cur_tile_size_M * cur_tile_size_K
                        elif buffer_state_act == "store all" and n == 0:
                            stats.reads['dram'] += cur_tile_size_M * cur_tile_size_K
                            stats.writes['g_act'] += cur_tile_size_M * cur_tile_size_K
                    if not weight_stored_in_buffer:
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
                    stats.reads['g_act'] += cur_tile_size_M * cur_tile_size_K
                    stats.writes['l_act'] += cur_tile_size_M * cur_tile_size_K
                    
                    preprocess_act, cur_cycles = self.find_reuse(cur_act)
                    compute_cycles += torch.sum(preprocess_act != 0).item()
                    stats.reads['l_act'] += cur_tile_size_M * cur_tile_size_K
                    stats.reads['l_wgt'] += torch.sum(preprocess_act != 0).item() * cur_tile_size_N * operator.weight_tensor.nbits
                    # find the row in preprocess_act that is all zero, if all zero originally, no cycles needed, if not, need one cycle
                    nnz_each_row = torch.sum(preprocess_act != 0, dim=-1)
                    nnz_each_row_ori = torch.sum(cur_act != 0, dim=-1)
                    all_zero_row = nnz_each_row == 0
                    all_zero_row_ori = nnz_each_row_ori == 0
                    compute_cycles += torch.sum(all_zero_row).item() - torch.sum(all_zero_row_ori).item()

                    preprocess_cycles += cur_cycles
                    orginial_compute_cycles += torch.sum(cur_act != 0).item()
                    stats.num_ops += torch.sum(preprocess_act != 0).item() * cur_tile_size_N

                    # write results to partial sum buffer
                    stats.reads['g_psum'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits
                    stats.writes['g_psum'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits
                    # if k == tile_num_K - 1:
                    #     stats.reads['g_psum'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits
                    #     stats.writes['g_psum'] += cur_tile_size_M * cur_tile_size_N * operator.output_tensor.nbits

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

        if operator.attention_type == 'spikformer':
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

        elif operator.attention_type == 'SDT':
            init_mem_access = 0
            if not spike_stored_in_buffer:
                init_mem_access += operator.sequence_length * operator.dim_per_head * 3
                stats.reads['dram'] += operator.act_q_tensor.sparse_map.numel() * 3
            else:
                stats.reads['g_act'] += operator.act_q_tensor.sparse_map.numel() * 3

            stats.writes['l_act'] += operator.act_q_tensor.sparse_map.numel() * 3


            num_op = operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head * operator.sequence_length

            stats.compute_cycles += num_op // self.accelerator.adder_array_size
            stats.reads['l_act'] += operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head * operator.sequence_length * 2
            stats.writes['l_act'] += operator.time_steps * operator.batch_size * operator.num_head * operator.dim_per_head
            lif = LIFNeuron('lif_attn', operator.dim_per_head * operator.num_head, operator.batch_size, operator.time_steps)
            lif_stats, _ = self.run_LIF(lif, self.accelerator.adder_array_size, 1)
            stats.compute_cycles += max(0, lif_stats.total_cycles - stats.compute_cycles)
            stats.reads['l_act'] += num_op * self.accelerator.bit_operation_width * 2
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

        print(operator.name)
        print("total cycles: ", stats.total_cycles)
        print("compute cycles: ", stats.compute_cycles)
        if operator.attention_type == 'spikformer':
            out_spike_stored_in_buffer = False

        return stats, out_spike_stored_in_buffer
    
    def optimize_attention(self, act_k: torch.Tensor, act_v: torch.Tensor, operator: Attention, eq_sequence_length: int, eq_dim_per_head: int):

        act_k = act_k.reshape([operator.time_steps, operator.batch_size, eq_sequence_length, operator.num_head, eq_dim_per_head]).permute(0, 1, 3, 4, 2).contiguous()
        act_v = act_v.reshape([operator.time_steps, operator.batch_size, eq_sequence_length, operator.num_head, eq_dim_per_head]).permute(0, 1, 3, 4, 2).contiguous()
        act_k = act_k.reshape(act_k.shape[0], act_k.shape[1], act_k.shape[2], act_k.shape[3], -1, self.accelerator.bit_operation_width)
        act_v = act_v.reshape(act_v.shape[0], act_v.shape[1], act_v.shape[2], act_v.shape[3], -1, self.accelerator.bit_operation_width)
        k_row_nnz = torch.sum(act_k != 0, dim=-1).permute(0, 1, 2, 4, 3)
        v_row_nnz = torch.sum(act_v != 0, dim=-1).permute(0, 1, 2, 4, 3)
        k_nonzero_row = torch.sum(k_row_nnz != 0, dim=-1)
        v_nonzero_row = torch.sum(v_row_nnz != 0, dim=-1)

        optimized_computation = torch.sum(k_nonzero_row * v_nonzero_row).item()
        return optimized_computation

    
    def find_reuse(self, act: torch.Tensor):
        cycles = 0
        cycles += act.shape[0] // self.accelerator.num_popcnt # do popcnt to all rows
        preprocessed_act = act.clone()
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

            cycles += 1 # find all subset of this row in CAM
            # find the largest subset through look up the popcnt result

            if torch.sum(is_real_subset) == 0:
                # cycles += 1 # if no subset, search for next begin point in CAM
                continue
            subset_row = act[is_real_subset]
            subset_row_nnz = torch.sum(subset_row != 0, dim=-1)
            max_subset_size = torch.max(subset_row_nnz).item()
            max_subset = subset_row[torch.argmax(subset_row_nnz)]
            if max_subset_size > 1:
                preprocessed_act[i] = torch.logical_xor(preprocessed_act[i], max_subset)
            # else:
                # cycles += 1 # search the next begin point in CAM

        return preprocessed_act, cycles

    
    def find_largest_subset(self, operator: FC):
        input_shape = operator.activation_tensor.shape
        input_shape = [np.prod(input_shape[:-1]), input_shape[-1]]
        input_tensor = operator.activation_tensor.sparse_map.reshape(input_shape)
        # a torch tensor, dtype = bool
        tile_size_M = 256
        tile_size_K = 16
        total_nnzs = torch.sum(input_tensor != 0).item()
        reduced_nnz = total_nnzs
        for m in range(0, input_shape[0], tile_size_M):
            for k in range(0, input_shape[1], tile_size_K):
                cur_tile_size_M = min(tile_size_M, input_shape[0] - m)
                cur_tile_size_K = min(tile_size_K, input_shape[1] - k)
                cur_tensor = input_tensor[m:m+cur_tile_size_M, k:k+cur_tile_size_K]
                for i in range(cur_tile_size_M):
                    # find the largest subset of each row
                    cur_row = cur_tensor[i]
                    nnz = torch.sum(cur_row != 0).item()
                    if nnz < 2:
                        continue
                    # if A is subset of B, then A & B = A
                    and_result = torch.logical_and(cur_row, cur_tensor)
                    equalities = torch.eq(and_result, cur_tensor)
                    is_subset = torch.all(equalities, dim=-1)

                    equalities = torch.eq(cur_row, cur_tensor)
                    is_equal = torch.all(equalities, dim=-1)

                    is_bigger_index = torch.arange(cur_tile_size_M) >= i

                    # if A = B, then only reuse once
                    is_excluded = torch.logical_and(is_equal, is_bigger_index)
                    is_real_subset = torch.logical_and(is_subset, ~is_excluded)
                    if torch.sum(is_real_subset) == 0:
                        continue
                    subset_row = cur_tensor[is_real_subset]
                    subset_row_nnz = torch.sum(subset_row != 0, dim=-1)
                    max_subset = torch.max(subset_row_nnz).item()

                    # if exist subset and the size is larger than 1, it can be reused
                    if max_subset > 1:
                        if nnz == max_subset:
                            reduced_nnz -= max_subset - 1
                        else:
                            reduced_nnz -= max_subset
        print("preprocessed nnz: ", reduced_nnz)
        print("total nnz: ", total_nnzs)
        print("percentage: ", reduced_nnz / total_nnzs)
        return reduced_nnz, total_nnzs

                    
if __name__ == '__main__':
    # Adder 8 bit * 128, MAC 8 bit * 32, Bitwise 32 bit * 4

    accelerator = Accelerator(num_PE=8, sram_size={'wgt': 131072, 'act': 262144, 'psum': 64, 'out': 64}, adder_array_size=128, LIF_array_size=32)
    nn = create_network('SDT', 'sdt_cifar10.pkl')
    # nn = nn[10:]
    sim = Simulator(accelerator, nn)
    sim.run_simulation()
    # operator = nn[12]
    # sim.run_attention(operator, spike_stored_in_buffer=False)
    # sim.find_common_sequence(operator)
    # pre_nnzs = []
    # total_nnzs = []
    # for operator in nn:
    #     if isinstance(operator, Conv2D):
    #         eq_fc = conv2d_2_fc(operator)
    #         print(eq_fc.name)
    #         pre_nnz, total_nnz = sim.find_largest_subset(eq_fc)
    #     if isinstance(operator, FC):
    #         print(operator.name)
    #         pre_nnz, total_nnz = sim.find_largest_subset(operator)
    #     else:
    #         continue
    #     pre_nnzs.append(pre_nnz)
    #     total_nnzs.append(total_nnz)
    
    # print("preprocessed nnzs: ", sum(pre_nnzs))
    # print("total nnzs: ", sum(total_nnzs))
    # print("percentage: ", sum(pre_nnzs) / sum(total_nnzs))

    # fc = nn[10]
    # # create a torch tensor, that is a upper triangular matrix
    # b = torch.rand(10, 10)
    # # # uppder triangular
    # b = torch.triu(b, diagonal=1)
    # # cut half of the row
    # b = b[:5]
    # # b = torch.eye(100)
    # b = b.to(torch.bool)
    # cycles, new_b = sim.find_reuse(b)
    # print(cycles)

    # fc.activation_tensor.sparse_map = b
    # fc.activation_tensor.shape = b.shape
    # sim.find_largest_subset(fc)


