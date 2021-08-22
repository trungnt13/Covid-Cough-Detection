import glob
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
import zipfile
from dataclasses import dataclass

import torch
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import soundfile
import torchaudio
from six import string_types
from tqdm import tqdm

# ===========================================================================
# Constants
# ===========================================================================
SEED = 1
SAMPLE_RATE = 8000
# path to the downloaded *.zip
ROOT_PATH = Path('/mnt/sdb1/covid_data')

ZIP_FILES = dict(
  # warmup
  train=os.path.join(ROOT_PATH, 'aicv115m_public_train.zip'),
  pub_test=os.path.join(ROOT_PATH, 'aicv115m_public_test.zip'),
  pri_test=os.path.join(ROOT_PATH, 'aicv115m_private_test.zip'),
  # final
  final_train=os.path.join(ROOT_PATH, 'aicv115m_final_public_train.zip'),
  extra_train=os.path.join(ROOT_PATH, 'aicv115m_extra_public_1235samples.zip'),
  final_pub_test=os.path.join(ROOT_PATH, 'aicv115m_final_public_test.zip'),
  # fpri_test0=os.path.join(ROOT_PATH, 'aicv115m_final_private_test.zip'),
)

_dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def dev() -> torch.device:
  return _dev


# extra_train
# unknown    1203
# 1            21
# 0            11

@dataclass()
class Config:
  # batch_size
  bs: int = 64


# ===========================================================================
# Zip extract
# ===========================================================================

# === 1. extract the zip file
def _extract_zip():
  final_path = dict()
  for key, path in ZIP_FILES.items():
    name = os.path.basename(path).split('.')[0]
    abspath = list(ROOT_PATH.rglob(f'{name}.zip'))[0]
    outpath = os.path.join(ROOT_PATH, name)
    if not os.path.exists(outpath):
      with zipfile.ZipFile(abspath, mode='r') as f:
        namelist = [i for i in f.namelist() if '__MACOSX' not in i]
        print(f'Extract {abspath} to {outpath} ({len(namelist)} files)')
        f.extractall(ROOT_PATH, namelist)
    final_path[key] = outpath
  return final_path


PATH = _extract_zip()

# === cache and save path
CACHE_PATH = os.path.join(ROOT_PATH, 'cache')
SAVE_PATH = os.path.join(ROOT_PATH, 'results')
for path in [CACHE_PATH, SAVE_PATH]:
  if not os.path.exists(path):
    os.makedirs(path)

# uuid  subject_gender  subject_age   assessment_result   file_path
META_DATA = (lambda: {
  key: {
    os.path.basename(csv_file).replace('.csv', ''):
      pd.read_csv(csv_file.absolute(), sep=',', header=0)
    for csv_file in Path(val).glob('*.csv')}
  for key, val in PATH.items()})()


def _pos_weight():
  counts = list(META_DATA['final_train'].values())[
    0].assessment_result.value_counts()
  return counts[0] / counts[1]


POS_WEIGHT = 1.2 * _pos_weight()

# all wav files
WAV_FILES = {key: [str(i.absolute()) for i in Path(val).rglob('*.wav')
                   if '__MACOSX' not in str(i)]
             for key, val in PATH.items()}
for k, v in WAV_FILES.items():
  print(k, len(v), 'files')


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
             end: float = 1.0,
             seed: int = 1) -> Path:
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
  rand = np.random.RandomState(seed=seed)
  rand.shuffle(data)
  n = len(data)
  start = int(n * start)
  end = int(n * end)
  data = data[start:end]
  data = dict(data)
  # === 4. save to JSON
  path = os.path.join(CACHE_PATH,
                      f'{partition}_{start:g}_{end:g}_{seed:d}.json')
  with open(path, 'w') as f:
    json.dump(data, f)
  return Path(path)
