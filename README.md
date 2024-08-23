# Prosperity: Accelerating Spiking Neural Networks via Product Sparity

This repository is the official implementation of Prosperity: Accelerating Spiking Neural Networks via Product Sparity.

## Requirements

- Python == 3.10

## Setup

clone the repository and install the requirements

```bash
conda create -n prosperity python=3.10 -y
conda activate prosperity
pip install -r requirements.txt
git submodule update --init
```

install prosperity CUDA kernel
```bash
cd kernel
python setup.py install
```

Usage
```bash
python simulator.py
```