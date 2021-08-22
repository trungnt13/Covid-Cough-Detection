export COVID_PATH="/mnt/sdb1/covid_data"
export COVID_DEVICE="cuda:0"

# all arguments is defined in config.py Config
# careful overwrite will delete everything in the exist folder
python cough/train.py \
  -model wav2vec_en \
  -task covid \
  -oversampling True \
  -bs 16 \
  -pos_weight_rescale 0.2 \
  -random_cut 5 \
  -lr 0.0001 \
  -epochs 1000 \
  -ncpu 4 \
  --overwrite

