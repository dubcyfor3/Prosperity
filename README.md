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

### Run Prosperity and baselines simulation

```bash
./run_simulator_all.sh
```

This script will run the simulation of Prosperity and baselines on all of the models and datasets shown in Figure 8 in the paper. 
It is expected to take 15 minutes to finish with CUDA.
It will output the end to end runtime and energy consumption of each accelerator and on each dataset.
For Prosperity, it will also output the bit density and product density statistics in Figure 11 in the paper.

### Run Buffer Area Evaluation

```bash
python cacti.py
```

This program call the CACTI to evaluate the buffer area of Prosperity.

### ProSparsity Visualization

```bash
python sparse_analysis.py
```

To visualize the benefit of ProSparsity, this program discover product sparsity on a layer in spikformer model and visualize the submatrix of activation in bit sparsity and product sparsity.

### Design space exploration

Two key design choices in Prosperity, tile size on M dimension and K dimension, can be adjusted by specifying the tile_size_M and tile_size_K in input argument. 
However, the CUDA kernel is designed for default tile size 256x16. 
Therefore, the DSE can only be run on the CPU with longger simulation time.
An example for DSE is shown below.

```bash
python simulator --type Prosperity --tile_size_M 128 --tile_size_K 16
```