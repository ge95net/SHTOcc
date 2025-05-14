#!/usr/bin/env bash

module load anaconda/2024.02
module unload cuda/12.1
module load cuda/11.8
module unload cudnn/9.1.0_cu12x
module load cudnn/8.8.0_cu11x
module load gcc/9.3.0
# source activate b2d_zoo
source activate occformer

export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL


export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29504}


python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox
