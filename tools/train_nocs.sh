#!/usr/bin/env bash

PYTHON=${PYTHON:-"python"}

echo 'Training network heads'
$PYTHON tools/train.py configs/nocs/nocs_r50_fpn_1x_stage1.py --gpus=1

echo 'Training Resnet layer 4+'
$PYTHON tools/train.py configs/nocs/nocs_r50_fpn_1x_stage2.py --gpus=1

echo 'Training Resnet layer 3+'
$PYTHON tools/train.py configs/nocs/nocs_r50_fpn_1x_stage3.py --gpus=1
