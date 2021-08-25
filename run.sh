#!/bin/bash
export COVID_PATH="/mnt/sdb1/covid_data"
export COVID_DEVICE="cuda:0"
# If you use pretrained wav2vec2, SR=16000
export COVID_SR=16000
export SEED=1
export DATA_SEED=1

# main training
MODEL=simple_xvec
TASK="covid"
PREFIX="cut15"
ARGS='0.05,0.05'
BS=25

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder
python cough/train.py \
  -model $MODEL \
  -model_args $ARGS \
  -prefix $PREFIX \
  -task $TASK \
  -oversampling True \
  -bs $BS \
  -pos_weight_rescale 0.5 \
  -random_cut 15 \
  -lr 0.0005 \
  -epochs 1000 \
  -ncpu 4 \
  --overwrite

# pseudolabel
python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task pseudolabel \
  -bs $BS

# eval
python cough/train.py \
  -model $MODEL \
  -model_args $ARGS \
  -prefix $PREFIX \
  -task covid \
  -bs $BS \
  --eval

