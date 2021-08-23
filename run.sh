export COVID_PATH="/mnt/sdb1/covid_data"
export COVID_DEVICE="cuda:0"

# Config
MODEL=domain_xvec
BS=32

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder
python cough/train.py \
  -model $MODEL \
  -task covid \
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
#  -task covid \
#  -bs 16 \
#  --eval

