export COVID_PATH="/mnt/sdb1/covid_data"
export COVID_DEVICE="cuda:0"
# If you use pretrained wav2vec2, SR=16000
export COVID_SR=16000
export SEED=1
export DATA_SEED=1

## Contrastive learning
# Config
MODEL=contrastive_xvec
TASK="contrastive"
PREFIX="contr"
BS=40

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder
python cough/train.py \
  -model $MODEL \
  -prefix $PREFIX \
  -task $TASK \
  -bs $BS \
  -dropout 0.5 \
  -random_cut 15 \
  -lr 0.0001 \
  -epochs 10000 \
  -ncpu 4 \
  --overwrite
