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

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    ./tools/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}

# #!/usr/bin/env bash

# CONFIG=$1
# GPUS=$2
# PORT=${PORT:-28509}

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
