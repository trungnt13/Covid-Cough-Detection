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

BSpretrain=5
BSfinetune=16

## Contrastive learning
# Config
MODEL=contrastive_ecapa
TASK="contrastive"
PREFIX="contr_cut8"
CUT=8
NCPU=8

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder

python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task $TASK \
  -bs $BSpretrain \
  -dropout 0.5 \
  -random_cut $CUT \
  -lr 0.0005 \
  -epochs 10000 \
  -steps_priming 200 \
  -ncpu $NCPU \
  -pseudolabel True \
  --overwrite

## fine tuning

python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task covid \
  -bs $BSfinetune \
  -dropout 0.5 \
  -random_cut $CUT \
  -label_noise 0.15 \
  -lr 0.0008 \
  -epochs 1000 \
  -steps_priming 1000 \
  -ncpu $NCPU \
  -pseudolabel True \
  -monitor val_f1 \
  -load val_loss

# evaluating
python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task covid \
  -bs $BSfinetune \
  -load val_f1 \
  --eval

