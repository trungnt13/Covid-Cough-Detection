export COVID_PATH="/mnt/sdb1/covid_data"
export COVID_DEVICE="cuda:0"

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder
python cough/train.py \
  -model simple_ecapa \
  -task covid \
  -oversampling True \
  -bs 32 \
  -pos_weight_rescale 0.5 \
  -random_cut -1 \
  -lr 0.0001 \
  -epochs 1000 \
  -ncpu 4 \
  --overwrite

