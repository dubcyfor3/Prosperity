from utils import *

dram_enenrgy_per_bit = 17.97 # pJ

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
        'Eyeriss': 1128.4,
        'SATO': 150.88,
        'PTB': 796.4,
        'MINT': 317.1,
    }

    mem_access_ratio_dict = {   # derived from papers
        'Prosperity': 1.00,
        'Eyeriss': 12.3,
        'SATO': 6.42,
        'PTB': 5.76,
        'MINT': 1.48,
    }

    dram_enenrgy_per_bit = 18.11 # pJ, derived from DRAMsim3

    processing_time = stats.total_cycles / (500 * 1000 * 1000) # in seconds
    on_chip_power = on_chip_power_dict[type] # in mW
    on_chip_energy = on_chip_power * processing_time # in mJ
    
    dram_access = read_position(file_name='artifact_eval/mem.csv', column_name=model, row_name="mem_access")
    dram_access *= mem_access_ratio_dict[type]
    dram_energy = dram_enenrgy_per_bit * dram_access * 1e-9 # in mJ

    return on_chip_energy + dram_energy

if __name__ == '__main__':
    get_total_energy(Stats(), 'Prosperity', 'spikformer_cifar100')