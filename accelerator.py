from cacti import CactiSweep

class Accelerator:
    def __init__(self, type, adder_array_size, LIF_array_size, tile_size_M, tile_size_K, product_sparsity=True, dense=False, tree_manage_type=2, mem_if_width=1024):
        self.type = type
        self.num_popcnt = 8
        self.adder_array_size = adder_array_size  # tile size N
        self.LIF_array_size = LIF_array_size
        self.multiplier_array_size = 32
        self.num_exp = 8
        self.num_div = 1
        self.SpMM_tile_size_M = tile_size_M
        self.SpMM_tile_size_K = tile_size_K
        self.mem_if_width = mem_if_width
        self.tech_node = 0.028
        self.sram_size = {}
        self.sram_size['wgt'] = self.SpMM_tile_size_K * self.adder_array_size * 8    # global buffer ori 16 * 128 * 8
        self.sram_size['act'] = self.SpMM_tile_size_M * self.SpMM_tile_size_K * 1    # global buffer ori 16 * 256
        self.sram_size['out'] = self.SpMM_tile_size_M * self.adder_array_size * 8       # global buffer 258 * 16

        self.product_sparsity = product_sparsity
        self.dense = dense
        self.tree_manage_type = tree_manage_type # 0: store tree in adj 1: search tree through prefix 2: sorting based tree

    def get_sram_stats(self):
        cacti_sweep = CactiSweep()
        buffer_cfg = {'block size (bytes)': self.adder_array_size, 
                            'size (bytes)': self.SpMM_tile_size_M * self.adder_array_size, 
                            'technology (u)': self.tech_node}
        
        output_dict = {}
        output_dict['area'] = cacti_sweep.get_data_clean(buffer_cfg)['area_mm^2'].values[0] * 2 # 2 for ping-pong buffer
        output_dict['leak_power'] = cacti_sweep.get_data_clean(buffer_cfg)['leak_power_mW'].values[0] * 2
        output_dict['read_energy'] = cacti_sweep.get_data_clean(buffer_cfg)['read_energy_nJ'].values[0]
        output_dict['write_energy'] = cacti_sweep.get_data_clean(buffer_cfg)['write_energy_nJ'].values[0]

        return output_dict


