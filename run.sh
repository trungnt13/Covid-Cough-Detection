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
MODEL=domain_xvec
TASK="covid"
PREFIX="cut8"
ARGS='0.1'
BS=50
NCPU=8

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder

python cough/train.py \
  -model $MODEL \
  -model_args $ARGS \
  -prefix $PREFIX \
  -task $TASK \
  -oversampling True \
  -bs $BS \
  -label_noise 0.15 \
  -pos_weight_rescale 0.5 \
  -random_cut 8 \
  -lr 0.0008 \
  -epochs 1000 \
  -steps_priming 1000 \
  -ncpu $NCPU \
  -pseudolabel True \
  --overwrite

## pseudolabel

#python cough/train.py \
#  -model $MODEL \
#  -prefix $PREFIX \
#  -task pseudolabel \
#  -bs $BS


## eval

#python cough/train.py \
#  -model $MODEL \
#  -model_args $ARGS \
#  -prefix $PREFIX \
#  -task covid \
#  -bs $BS \
#  --eval

