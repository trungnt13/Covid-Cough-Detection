#!/bin/bash
export COVID_PATH="/mnt/sdb1/covid_data"
if [ ! -d "$COVID_PATH" ]
then
  export COVID_PATH="/home/trung/covid_data"
fi
export COVID_DEVICE="cuda:0"
# If you use pretrained wav2vec2, SR=16000
export COVID_SR=16000
export SEED=1
export DATA_SEED=1

# main training
MODEL=simple_gender
TASK="gender"
PREFIX="cut8"
BS=32
NCPU=8

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder
python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task $TASK \
  -oversampling True \
  -pos_weight_rescale 1.0 \
  -bs $BS \
  -random_cut 8 \
  -lr 0.0005 \
  -epochs 1000 \
  -steps_priming 300 \
  -ncpu $NCPU \
  --overwrite

## eval
#python cough/train.py \
#  -model $MODEL \
#  -model_args $ARGS \
#  -prefix $PREFIX \
#  -task covid \
#  -bs $BS \
#  --eval

