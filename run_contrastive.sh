#!/bin/bash


echo "--------------";
echo "Path                  : $1";
echo "Device                : $2";
echo "Model                 : $3";
echo "Batch size (pretrain) : $4";
echo "Batch size (finetune) : $5";
echo "Load for finetune     : $6";
echo "SEED                  : $7";
echo "--------------";

export COVID_PATH="$1"
if [ ! -d "$COVID_PATH" ]
then
  export COVID_PATH="/home/trung/covid_data"
fi
export COVID_DEVICE="$2"
export COVID_SR=16000
export SEED=1
export DATA_SEED="$7"

NCPU=6

## Contrastive learning
# Config
MODEL=$3

BSpretrain=$4
BSfinetune=$5

TASK="contrastive"
ARGS="0.1"
CUT=15
PREFIX="ctrs_cut$CUT_$6"

EPOCH1=10000 # pretrain
EPOCH2=1000 # finetune

POS_WEIGHT=0.6

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder

python cough/train.py \
  -model $MODEL \
  -model_args $ARGS \
  -prefix $PREFIX \
  -task $TASK \
  -bs $BSpretrain \
  -dropout 0.3 \
  -random_cut $CUT \
  -lr 0.0008 \
  -epochs $EPOCH1 \
  -steps_priming 200 \
  -ncpu $NCPU \
  --pseudolabel \
  --overwrite

## fine tuning

python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task covid \
  -bs $BSfinetune \
  -dropout 0.3 \
  -random_cut $CUT \
  -label_noise 0.1 \
  -pos_weight_rescale $POS_WEIGHT \
  -lr 0.0008 \
  -epochs $EPOCH2 \
  -steps_priming 1000 \
  -ncpu $NCPU \
  -monitor val_f1 \
  -load $6 \
  --oversampling \
  --pseudolabel

# evaluating

python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task covid \
  -bs $BSfinetune \
  -load val_f1 \
  --eval

