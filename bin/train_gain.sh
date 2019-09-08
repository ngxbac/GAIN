#!/usr/bin/env bash

#export CUDA_VISIBLE_DEVICES=2,3
RUN_CONFIG=config_gain.yml


log_name=gain_ip102
LOGDIR=/media/ngxbac/DATA/logs_gain/${log_name}/
catalyst-dl run \
    --config=./configs/${RUN_CONFIG} \
    --logdir=$LOGDIR \
    --out_dir=$LOGDIR:str \
    --monitoring_params/name=${log_name}:str \
    --stages/stage0/callbacks_params/heatmap_saver/outdir=$LOGDIR/heatmaps/:str \
    --verbose