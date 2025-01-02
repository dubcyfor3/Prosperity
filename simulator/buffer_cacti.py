import subprocess
import pandas
import os
import re
import json

class CactiSweep(object):
    def __init__(self, bin_file='../cacti/cacti', csv_file='cacti_stats.csv', default_json='./sram_config.json'):
        if not os.path.isfile(bin_file):
            print("Can't find binary file {}. Please clone and compile cacti first".format(bin_file))
            self.bin_file = None
        else:
            self.bin_file = os.path.abspath(bin_file)
        self.csv_file = os.path.abspath(os.path.join(os.path.dirname(__file__), csv_file))
        self.default_dict = json.load(open(default_json))
        self.cfg_file = os.path.join(os.path.dirname(os.path.abspath(self.csv_file)), 'cacti.cfg')

        output_dict = {
                'Access time (ns)': 'access_time_ns',
                'Total dynamic read energy per access (nJ)': 'read_energy_nJ',
                'Total dynamic write energy per access (nJ)': 'write_energy_nJ',
                'Total leakage power of a bank (mW)': 'leak_power_mW',
                'Total gate leakage power of a bank (mW)': 'gate_leak_power_mW',
                'Cache height (mm)': 'height_mm',
                'Cache width (mm)': 'width_mm',
                'Cache area (mm^2)': 'area_mm^2',
                }
        cols = list(self.default_dict)
        cols.extend(output_dict.keys())
        self._df = pandas.DataFrame(columns=cols)

    def update_csv(self):
        self._df = self._df.drop_duplicates()
        self._df.to_csv(self.csv_file, index=False)

    def _create_cfg(self, cfg_dict, filename):
        with open(filename, 'w') as f:
            cfg_dict['output/input bus width'] = cfg_dict['block size (bytes)'] * 8
            for key in cfg_dict:
                if cfg_dict[key] is not None:
                    f.write('-{} {}\n'.format(key, cfg_dict[key]))

    def _parse_cacti_output(self, out):
        output_dict = {
                'Access time (ns)': 'access_time_ns',
                'Total dynamic read energy per access (nJ)': 'read_energy_nJ',
                'Total dynamic write energy per access (nJ)': 'write_energy_nJ',
                'Total leakage power of a bank (mW)': 'leak_power_mW',
                'Total gate leakage power of a bank (mW)': 'gate_leak_power_mW',
                # 'Cache height (mm)': 'height_mm',
                # 'Cache width (mm)': 'width_mm',
                # 'Cache area (mm^2)': 'area_mm^2',
                'Cache height x width (mm)': 'height_mm',
                }
        parsed_results = {}
        for line in out:
            line = line.rstrip()
            line = line.lstrip()
            if line:
                for o in output_dict:
                    key = output_dict[o]
                    o = o.replace('(', '\(')
                    o = o.replace(')', '\)')
                    o = o.replace('^', '\^')
                    regex = r"{}\s*:\s*([\d\.]*)".format(o)
                    m = re.match(regex, line.decode('utf-8'))
                    if m:
                        parsed_results[key] = m.groups()[0]
                        if key == "height_mm":
                            regex = r"{}\s*x\s*([\d\.]*)".format(o + ": " + m.groups()[0])
                            m = re.match(regex, line.decode('utf-8'))
                            parsed_results['width_mm'] = m.groups()[0]
        return parsed_results

    def _run_cacti(self, index_dict):
        """
        Get data from cacti
        """
        assert self.bin_file is not None, 'Can\'t run cacti, no binary found. Please clone and compile cacti first.'
        cfg_dict = self.default_dict.copy()
        cfg_dict.update(index_dict)
        self._create_cfg(cfg_dict, self.cfg_file)
        args = (self.bin_file, "-infile", self.cfg_file)
        print(args)
        popen = subprocess.Popen(args, stdout=subprocess.PIPE, cwd=os.path.dirname(self.bin_file))
        popen.wait()
        output = popen.stdout
        cfg_dict.update(self._parse_cacti_output(output))
        return cfg_dict

    def locate(self, index_dict):
        self._df = self._df.drop_duplicates()
        data = self._df

        for key in index_dict:
            data = data.loc[data[key] == index_dict[key]]
        return data

    def get_data(self, index_dict):
        data = self.locate(index_dict)
        if len(data) == 0:
            print('running cacti')
            row_dict = index_dict.copy()
            row_dict.update(self._run_cacti(index_dict))
            row_dict["area_mm^2"] = float(row_dict["height_mm"]) * float(row_dict["width_mm"])
            if not self._df.empty:
                    self._df = pandas.concat([self._df, pandas.DataFrame([row_dict])], ignore_index=True)
            else:
                self._df = pandas.DataFrame([row_dict])
            self.update_csv()
            return self.locate(index_dict)
        else:
            return data

    def get_data_clean(self, index_dict):
        data = self.get_data(index_dict)
        cols = [
                'size (bytes)',
                'block size (bytes)',
                'access_time_ns',
                'read_energy_nJ',
                'write_energy_nJ',
                'leak_power_mW',
                'gate_leak_power_mW',
                'height_mm',
                'width_mm',
                'area_mm^2',
                'technology (u)',
                ]
        return data[cols]
    
def get_buffer_area(tech_node, buffer_size, block_size):
    buffer_sweep_data = CactiSweep()
    cfg_dict = {'block size (bytes)': block_size, 'size (bytes)': buffer_size, 'technology (u)': tech_node}
    buffer_area = buffer_sweep_data.get_data_clean(cfg_dict)['area_mm^2'].item()
    return buffer_area

def get_buffer_power_energy(tech_node, buffer_size, block_size):
    buffer_sweep_data = CactiSweep()
    cfg_dict = {'block size (bytes)': block_size, 'size (bytes)': buffer_size, 'technology (u)': tech_node}
    buffer_read_energy = float(buffer_sweep_data.get_data_clean(cfg_dict)['read_energy_nJ'].item()) # per buffer access
    buffer_write_energy = float(buffer_sweep_data.get_data_clean(cfg_dict)['write_energy_nJ'].item()) # per buffer access
    buffer_leak_power = float(buffer_sweep_data.get_data_clean(cfg_dict)['leak_power_mW'].item())
    return buffer_leak_power, buffer_read_energy, buffer_write_energy

if __name__ == "__main__":

    tech_node = 0.028
    act_buffer_config = {   # 8KB
        'buffer_size': 8192,
        'block_size': 4,
    }
    wgt_buffer_config = {   # 32KB
        'buffer_size': 32384,
        'block_size': 128,
    }
    out_buffer_0_config = {   # 96KB in total
        'buffer_size': 32384,
        'block_size': 128,
    }
    out_buffer_1_config = {
        'buffer_size': 65536,
        'block_size': 128,
    }

    act_buffer_area = get_buffer_area(tech_node, act_buffer_config['buffer_size'], act_buffer_config['block_size'])
    wgt_buffer_area = get_buffer_area(tech_node, wgt_buffer_config['buffer_size'], wgt_buffer_config['block_size'])
    out_buffer_0_area = get_buffer_area(tech_node, out_buffer_0_config['buffer_size'], out_buffer_1_config['block_size'])
    out_buffer_1_area = get_buffer_area(tech_node, out_buffer_1_config['buffer_size'], out_buffer_1_config['block_size'])
    act_buffer_power_static = get_buffer_power_energy(tech_node, act_buffer_config['buffer_size'], act_buffer_config['block_size'])
    wgt_buffer_power_static = get_buffer_power_energy(tech_node, wgt_buffer_config['buffer_size'], wgt_buffer_config['block_size'])
    out_buffer_0_power_static = get_buffer_power_energy(tech_node, out_buffer_0_config['buffer_size'], out_buffer_1_config['block_size'])
    out_buffer_1_power_static = get_buffer_power_energy(tech_node, out_buffer_1_config['buffer_size'], out_buffer_1_config['block_size'])
    buffer_access_energy_per_bit = (act_buffer_power_static[1] / (act_buffer_config['block_size'] * 8)+ 
                                    wgt_buffer_power_static[1] / (wgt_buffer_config['block_size'] * 8) +
                                    out_buffer_0_power_static[1] / (out_buffer_0_config['block_size'] * 8) + 
                                    out_buffer_1_power_static[1] / (out_buffer_1_config['block_size'] * 8) + 
                                    act_buffer_power_static[2] / (act_buffer_config['block_size'] * 8) +
                                    wgt_buffer_power_static[2] / (wgt_buffer_config['block_size'] * 8) +
                                    out_buffer_0_power_static[2] / (out_buffer_0_config['block_size'] * 8) + 
                                    out_buffer_1_power_static[2] / (out_buffer_1_config['block_size'] * 8)) / 8
    
    buffer_access_per_cycle = 128 + 256 # an estimated value

    dram_enenrgy_per_bit = 12.45 # pJ, derived from DRAMsim3
    dram_access = 140574720 # stats derived on spikformer cifar10
    time = 0.00374337 # stats derived on spikformer cifar10

    total_buffer_area = act_buffer_area + wgt_buffer_area + out_buffer_0_area + out_buffer_1_area
    total_buffer_static_power = act_buffer_power_static[0] + wgt_buffer_power_static[0] + out_buffer_0_power_static[0] + out_buffer_1_power_static[0]
    total_buffer_dynamic_power = buffer_access_per_cycle * buffer_access_energy_per_bit * 500 * 1000 * 1000 / 1e6
    dram_power = dram_enenrgy_per_bit * dram_access * 1e-9 / time
    print('Prosperity total buffer area: {} mm^2'.format(total_buffer_area))
    print('Prosperity total buffer static power: {} mW'.format(total_buffer_static_power))
    print('Prosperity total buffer dynamic power: {} mW'.format(total_buffer_dynamic_power))
    print('Prosperity total buffer power: {} mW'.format(total_buffer_static_power + total_buffer_dynamic_power))
    print('Prosperity total DRAM power: {} mW'.format(dram_power))