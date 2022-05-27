#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

python -m torch.distributed.launch --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
