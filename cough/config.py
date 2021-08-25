import json
import os
import pickle
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile
import torch
from joblib import Parallel, delayed
from six import string_types
from tqdm import tqdm

# ===========================================================================
# Constants
# ===========================================================================
SEED = int(os.environ.get('SEED', 1))
DATA_SEED = int(os.environ.get('DATA_SEED', 1))

SAMPLE_RATE = int(os.environ.get('COVID_SR', 8000))
# path to the downloaded *.zip
COVID_PATH = Path(os.environ.get('COVID_PATH', '/mnt/sdb1/covid_data'))
print(f'* Read Covid data at path COVID_PATH={COVID_PATH}')

ZIP_FILES = dict(
  # warmup
  # train=os.path.join(COVID_PATH, 'aicv115m_public_train.zip'),
  # pub_test=os.path.join(COVID_PATH, 'aicv115m_public_test.zip'),
  # pri_test=os.path.join(COVID_PATH, 'aicv115m_private_test.zip'),
  # final
  final_train=os.path.join(COVID_PATH, 'aicv115m_final_public_train.zip'),
  extra_train=os.path.join(COVID_PATH, 'aicv115m_extra_public_1235samples.zip'),
  final_pub_test=os.path.join(COVID_PATH, 'aicv115m_final_public_test.zip'),
  final_pri_test=os.path.join(COVID_PATH, 'aicv115m_final_private_test.zip'),
)

_dev = torch.device(os.environ.get("COVID_DEVICE", "cuda:0")
                    if torch.cuda.is_available() else "cpu")
print(f'* Running device COVID_DEVICE={_dev}')


def dev() -> torch.device:
  return _dev


# extra_train
# unknown    1203
# 1            21
# 0            11

@dataclass(frozen=False)
class Config:
  # batch_size
  bs: int = 32
  # rescale the positive weight BCE
  pos_weight_rescale: float = 0.5
  # dropout amount for everything
  dropout: float = 0.5
  # priming the classifier head first then fine-tuning the whole network
  steps_priming: int = 1000
  lr: float = 1e-4
  # exp: ExponentialLR
  # step: StepLR
  # cos: CosineAnnealingLR
  # cyc: CyclicLR
  scheduler: str = 'exp'
  gamma: float = 0.98
  lr_step: int = 100  # used for StepLR
  grad_clip: float = 0.0  # norm clipping
  epochs: int = 1000
  patience: int = 20
  label_noise: float = 0.1
  oversampling: bool = True
  ncpu: int = 4
  eval: bool = False
  overwrite: bool = False
  # randomly cut small segment of the audio file during training
  # if > 0, the duration in seconds if each segment
  random_cut: float = -1.
  # - 'covid': main system covid cough detection
  # - 'gender': train a gender classifier
  # - 'age': train an age classifier
  # - 'contrastive' : pretrain contrastive learning
  # - 'pseudolabel' : pseudo labeling all the dataset
  task: str = 'covid'
  # name of the model defined in models.py
  model: str = 'simple_xvec'
  # extra arguments for the model, e.g. 0.1,0.2,0.3
  model_args: str = ''
  # extra prefix for model identification
  prefix: str = ''
  # which metric for monitoring validation performance
  monitor: str = 'val_f1'
  # which model selected for evaluation
  top: int = 0
  # enable using pseudolabel
  pseudolabel: bool = False
  pseudosoft: bool = False
  pseudorand: bool = False


# ===========================================================================
# Zip extract
# ===========================================================================

# === 1. extract the zip file
def _extract_zip():
  final_path = dict()
  for key, path in ZIP_FILES.items():
    name = os.path.basename(path).split('.')[0]
    abspath = list(COVID_PATH.rglob(f'{name}.zip'))
    if len(abspath) == 0:
      print('Cannot find zip file at:', path)
      continue
    abspath = abspath[0]
    outpath = os.path.join(COVID_PATH, name)
    if not os.path.exists(outpath):
      with zipfile.ZipFile(abspath, mode='r') as f:
        namelist = [i for i in f.namelist() if '__MACOSX' not in i]
        print(f'Extract {abspath} to {outpath} ({len(namelist)} files)')
        f.extractall(COVID_PATH, namelist)
    final_path[key] = outpath
  return final_path


PATH = _extract_zip()

# === cache and save path
CACHE_PATH = os.path.join(COVID_PATH, 'cache')
SAVE_PATH = os.path.join(COVID_PATH, 'results')
PSEUDOLABEL_PATH = os.path.join(COVID_PATH, 'pseudolabel')

for path in [CACHE_PATH, SAVE_PATH, PSEUDOLABEL_PATH]:
  if not os.path.exists(path):
    os.makedirs(path)


# uuid  subject_gender  subject_age   assessment_result   file_path
def _meta_data():
  return {
    key: {
      os.path.basename(csv_file).replace('.csv', ''):
        pd.read_csv(csv_file.absolute(), sep=',', header=0)
      for csv_file in Path(val).glob('*.csv')}
    for key, val in PATH.items()}


META_DATA = _meta_data()


# get positive weight from 'final_train'
def _pos_weight():
  counts = META_DATA[
    'final_train'][
    'public_train_metadata'].assessment_result.value_counts()
  return counts[0] / counts[1]


POS_WEIGHT = _pos_weight()


# all wav files
def _wav_files():
  wavs = {key: [str(i.absolute())
                for i in Path(val).rglob('*.wav')
                if '__MACOSX' not in str(i)]
          for key, val in PATH.items()}
  for k, v in wavs.items():
    print(k, len(v), 'files')
  return wavs


WAV_FILES = _wav_files()


# extraction duration and sample rate

def _wav_meta():
  cache_path = os.path.join(CACHE_PATH, 'wav_meta')
  if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
      meta = pickle.load(f)
  else:
    meta = defaultdict(dict)

  def _extract(args):
    part, path = args
    y, sr = soundfile.read(path)
    return (part, path, max(y.shape) / sr, sr)

  updated = False
  for partition in WAV_FILES.keys():
    if partition not in meta:
      jobs = [(partition, i) for i in WAV_FILES[partition]]
      for part, path, dur, sr in tqdm(
          Parallel(n_jobs=3)(delayed(_extract)(j) for j in jobs),
          desc='Caching WAV metadata',
          total=len(jobs)):
        meta[part][path] = (dur, sr)
      updated = True
  if updated:
    with open(cache_path, 'wb') as f:
      pickle.dump(meta, f)
  return meta


# mapping: partition -> {path: (duration, sample_rate)}
WAV_META = _wav_meta()


# ===========================================================================
# Pre-processing
# ===========================================================================
def get_json(partition: str,
             start: float = 0.0,
             end: float = 1.0) -> Path:
  """path, gender, age, result

  result=-1 for test set

  Example
  -------
  ```
  JSON_TRAIN = get_json('train', start=0.0, end=0.8)
  JSON_VALID = get_json('train', start=0.8, end=1.0)
  ```
  """
  wav = WAV_FILES[partition]
  wav_meta = WAV_META[partition]
  # === 1. prepare meta
  meta = defaultdict(dict)
  for k, tab in META_DATA[partition].items():
    tab: pd.DataFrame
    for _, row in tab.iterrows():
      meta[row['uuid']].update({
        i: eval(j) if isinstance(j, string_types) and '[' in j else j
        for i, j in row.items()})
  # === 2. load wav
  data = []
  for f in sorted(wav):
    name = os.path.basename(f)
    uuid = name.replace('.wav', '')
    row: dict = meta[uuid]
    dur, sr = wav_meta[f]
    row['duration'] = dur
    row['sr'] = sr
    data.append((uuid, dict(path=f, meta=row)))
  # === 3. shuffle and split
  rand = np.random.RandomState(seed=DATA_SEED)
  rand.shuffle(data)
  n = len(data)
  start = int(n * start)
  end = int(n * end)
  data = data[start:end]
  data = dict(data)
  # === 4. save to JSON
  path = os.path.join(CACHE_PATH,
                      f'{partition}_{start:g}_{end:g}_{DATA_SEED:d}.json')
  with open(path, 'w') as f:
    json.dump(data, f)
  return Path(path)
