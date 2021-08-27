#!/bin/bash
echo "--------------";
echo "Path      : $1";
echo "Device    : $2";
echo "Model     : $3";
echo "Batch size: $4";
echo "NCPU      : $5";
echo "SEED      : $6";
echo "--------------";

export COVID_PATH="$1"
if [ ! -d "$COVID_PATH" ]
then
  export COVID_PATH="/home/trung/covid_data"
fi
export COVID_DEVICE="$2"
export COVID_SR=16000
export SEED=1
export DATA_SEED="$6"

# main training
MODEL=$3
TASK="covid"
PREFIX="cut8"
BS=$4
NCPU=$5
EPOCH=1000


# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder

python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task $TASK \
  -bs $BS \
  -label_noise 0.2 \
  -pos_weight_rescale 0.5 \
  -random_cut 8 \
  -lr 0.0008 \
  -epochs $EPOCH \
  -steps_priming 1000 \
  -ncpu $NCPU \
  --oversampling \
  --overwrite

## pseudolabel (just call eval)

python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -bs $BS \
  --save_pseudo --eval
