from utils import *

def get_total_energy(stats: Stats, type: str, model) -> float:
    """
    Calculate the total energy consumed by the accelerator.
    
    Args:
    stats (Stats): Stats object containing the energy consumption details
    
    Returns:
    float: Total energy consumed in m Joules
    """

    on_chip_power_dict = {  # derived from papers
        'Prosperity': 446.5,
        'Eyeriss': 1410.5,
        'SATO': 319.5,
        'PTB': 982.0,
        'MINT': 396.3,
        'Stellar': 834.0,
    }

    mem_access_ratio_dict = {   # derived from papers
        'Prosperity': 1.00,
        'Eyeriss': 5.47,
        'SATO': 4.85,
        'PTB': 2.53,
        'MINT': 3.12,
        'Stellar': 1.23,
    }

    if type == 'A100':
        return get_total_energy_a100(stats, model)
    elif type not in on_chip_power_dict:
        raise ValueError("Invalid accelerator type: {}".format(type))
    else:
    
        dram_enenrgy_per_bit = 12.45 # pJ, derived from DRAMsim3

        if stats.total_cycles != None:
            processing_time = stats.total_cycles / (500 * 1000 * 1000) # in seconds
            on_chip_power = on_chip_power_dict[type] # in mW
            on_chip_energy = on_chip_power * processing_time # in mJ
        
            dram_access = read_position(file_name='../reference/mem_reference.csv', column_name=model, row_name="mem_access")
            dram_access *= mem_access_ratio_dict[type]
            dram_energy = dram_enenrgy_per_bit * dram_access * 1e-9 # in mJ
            total_energy = on_chip_energy + dram_energy
        else:
            total_energy = None
        

    return total_energy

def get_total_energy_a100(stats: Stats, model):
    
    total_power_dict = {
        'spikformer_cifar10': 96,
        'spikformer_cifar10dvs': 84,
        'spikformer_cifar100': 90,
        'sdt_cifar10': 91,
        'sdt_cifar10dvs': 81,
        'sdt_cifar100': 89,
        'spikebert_sst2': 132,
        'spikebert_mr': 184,
        'spikebert_sst5': 137,
        'spikingbert_sst2': 107,
        'spikingbert_qqp': 109,
        'spikingbert_mnli': 107,
    }
    if model not in total_power_dict:
        return None
    total_power = total_power_dict[model] * 1000 # in mW
    processing_time = stats.total_cycles / (500 * 1000 * 1000) # in seconds
    total_energy = total_power * processing_time # in mJ

    return total_energy

if __name__ == '__main__':
    get_total_energy(Stats(), 'Prosperity', 'spikformer_cifar100')