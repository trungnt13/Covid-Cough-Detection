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


## Contrastive learning
# Config
MODEL=contrastive_xvec
TASK="contrastive"
PREFIX="contr_cut10"
BS=24

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder
#python cough/train.py \
#  -model $MODEL \
#  -prefix $PREFIX \
#  -task $TASK \
#  -bs $BS \
#  -dropout 0.5 \
#  -random_cut 10 \
#  -lr 0.0005 \
#  -epochs 10000 \
#  -steps_priming 200 \
#  -ncpu 4 \
#  -pseudolabel True \
#  --overwrite

## fine tuning
BS=64

#python cough/train.py \
#  -model $MODEL \
#  -prefix $PREFIX \
#  -task covid \
#  -bs $BS \
#  -dropout 0.5 \
#  -random_cut 10 \
#  -lr 0.0005 \
#  -epochs 1000 \
#  -steps_priming 1000 \
#  -ncpu 4 \
#  -pseudolabel True

# evaluating
python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task covid \
  -bs $BS \
  --eval

