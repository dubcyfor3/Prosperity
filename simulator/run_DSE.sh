#!/bin/bash

# Run all configurations for the DSE

python simulator.py --type Prosperity --tile_size_M 256 --tile_size_K 4 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 256 --tile_size_K 8 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 256 --tile_size_K 32 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 256 --tile_size_K 64 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 256 --tile_size_K 128 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 32 --tile_size_K 16 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 64 --tile_size_K 16 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 128 --tile_size_K 16 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 256 --tile_size_K 16 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 512 --tile_size_K 16 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 1024 --tile_size_K 16 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --tile_size_M 2048 --tile_size_K 16 --use_cuda --dse_mode --output_dir ../dse
python simulator.py --type Prosperity --bit_sparsity --dse_mode --output_dir ../dse

python dse_post_process.py