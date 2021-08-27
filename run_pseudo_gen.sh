#!/bin/bash
echo "--------------";
echo "Path      : $1";
echo "Device    : $2";
echo "Batch size: $3";
echo "NCPU      : $4";
echo "SEED      : $5";
echo "--------------";

export COVID_PATH="$1"
if [ ! -d "$COVID_PATH" ]
then
  export COVID_PATH="/home/trung/covid_data"
fi
export COVID_DEVICE="$2"
export COVID_SR=16000
export SEED=1
export DATA_SEED="$5"

# main training
MODEL=simple_gender
TASK="gender"
PREFIX="cut8"
BS=$3
NCPU=$4

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder
python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task $TASK \
  -oversampling True \
  -pos_weight_rescale 0.6 \
  -bs $BS \
  -random_cut 8 \
  -lr 0.0008 \
  -epochs 1 \
  -steps_priming 300 \
  -ncpu $NCPU \
  --overwrite

## eval
python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task $TASK \
  -bs $BS \
  --eval

