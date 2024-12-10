# Prosperity: Accelerating Spiking Neural Networks via Product Sparity

This repository is the official implementation of HPCA 2025 paper, Prosperity: Accelerating Spiking Neural Networks via Product Sparity. It contains the cycle-accurate simulator for Prosperity architecture and implementation of several baseline accelerators.

## TL;DR
We propose a Spiking Neural Networks (SNNs) accelerator, **Prosperity**, that leverage a novel **product sparsity** paradigm to improve various SNNs efficiency.

## Usage

### Recommend Environment

- Python == 3.10
- GCC/G++ >= 11.3.0
- CUDA >= 12.1

We recommend using conda to manage the environment.


### Setup

Clone the repository and install the requirements, cloning may take a while due to the large size of the data in the repository.

```bash
conda create -n prosperity python=3.10 -y
conda activate prosperity
pip install -r requirements.txt
git submodule update --init
```

Install Prosperity CUDA kernel
```bash
cd kernels
python setup.py install
cd ..
```

Install CACTI
```bash
cd cacti
make
cd ..
```

### Repository Structure

- `simulator`: Cycle-accurate simulator for Prosperity and baselines
- `kernels`: CUDA kernel for fast simulation of Prosperity
- `data`: The activation matrix data for simulation
- `cacti`: CACTI for buffer area evaluation
- `reference`: Reference results for Prosperity and baselines

### Run Prosperity and baselines simulation (Figure 8) (15 minutes)

```bash
./run_simulator_all.sh
```

This script will run the simulation of Prosperity and baselines on all of the models and datasets shown in Figure 8 in the paper. 
It is expected to take 15 minutes to finish with CUDA.
It will output the end to end runtime and energy consumption of each accelerator and on each dataset.
The results will be stored in `output` folder, named as `time.csv` and `energy.csv`.
We also provide a reference file in `reference` folder. `time_reference.xlsx` and `energy_reference.xlsx` contain reference results and figure 8 in the paper.
For Prosperity, it will also output the bit density and product density statistics in Figure 11 in the paper.

### Design space exploration (Figure 7, without power and area stats) (1 hour 30 minutes)

For two key design choices in Prosperity, tile size on M dimension and K dimension, we provide a script that test different tile_size_M and tile_size_K configuration in Prosperity and organize this result into file `M_dse.csv` for M dimension exploration and file `K_dse.csv` for K dimension exploration.
File `M_dse_reference.xlsx` and `K_dse_reference.xlsx` in `reference` folder contain reference DSE results and figure 7 in the paper.
Notice that this script does not include power and area stats, since it is evaluated by synopsys design compiler, we are not able to release the script for this part under the license.


```bash
sh run_DSE.sh
```

### Run Sparsity Analysis (Figure 11) (7 minutes)

```bash
python simulator.py --type Prospeity --sparse_analysis_mode --use_cuda
```

The script record the bit density and product density of Prosperity on all the models and datasets in Figure 11 in the paper.
The results will be stored in file `density_analysis.csv`. The reference file is `density_analysis_reference.xlsx` in `reference` folder.
Notice that the output file does not contain the density stats for FS neuron since this result is directly derived from the baseline paper.

### Run Buffer Area Evaluation (Figure 10 buffer area stats) (1 second)

```bash
python cacti.py
```

This program call the CACTI to evaluate the buffer area of Prosperity. 
The output area should be identical to the value in left pie chart in Figure 10 in the paper.

<!-- ### ProSparsity Visualization

```bash
python sparse_analysis.py
```

To visualize the benefit of ProSparsity, this program discover product sparsity on a layer in spikformer model and visualize the submatrix of activation in bit sparsity and product sparsity. -->
