#!/bin/bash
module load anaconda/2024.02
module unload cuda/12.1
module load cuda/11.8
module unload cudnn/9.1.0_cu12x
module load cudnn/8.8.0_cu11x
module load gcc/9.3.0
# source activate b2d_zoo
source activate occformer

export PYTHONUNBUFFERED=1



MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v -e .

