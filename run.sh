export COVID_PATH="/mnt/sdb1/covid_data"
export COVID_DEVICE="cuda:0"
# If you use pretrained wav2vec2, SR=16000
export COVID_SR=16000

# Config
MODEL=contrastive_xvec
TASK="contrastive"
PREFIX="contr"
ARGS='0.05,0.05'
BS=4

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
  -random_cut 10 \
  -lr 0.0001 \
  -epochs 1000 \
  -ncpu 4 \
  --overwrite

#python cough/train.py \
#  -model $MODEL \
#  -model_args $ARGS \
#  -prefix $PREFIX \
#  -task covid \
#  -bs 16 \
#  --eval

