import glob
import json
import os
from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from six import string_types

SEED = 1
# path to the downloaded *.zip
ROOT_PATH = Path('/mnt/sdb1/covid_data')

ZIP_FILES = dict(
  # warmup
  train=os.path.join(ROOT_PATH, 'aicv115m_public_train.zip'),
  pub_test0=os.path.join(ROOT_PATH, 'aicv115m_public_test.zip'),
  pri_test0=os.path.join(ROOT_PATH, 'aicv115m_private_test.zip'),
  # final
  final_train=os.path.join(ROOT_PATH, 'aicv115m_final_public_train.zip'),
  final_pub_test0=os.path.join(ROOT_PATH, 'aicv115m_final_public_test.zip'),
  # fpri_test0=os.path.join(ROOT_PATH, 'aicv115m_final_private_test.zip'),
)


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

# all wav files
WAV_FILES = {key: [str(i.absolute()) for i in Path(val).rglob('*.wav')
                   if '__MACOSX' not in str(i)]
             for key, val in PATH.items()}
for k, v in WAV_FILES.items():
  print(k, len(v), 'files')


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
  # === 1. prepare meta
  meta = defaultdict(dict)
  for k, tab in META_DATA[partition].items():
    tab: pd.DataFrame
    for _, row in tab.iterrows():
      meta[row['uuid']].update({
        i: eval(j) if isinstance(j, string_types) and '[' in j else j
        for i, j in row.items() if i != 'uuid'})
  # === 2. load wav
  data = []
  for f in sorted(wav):
    name = os.path.basename(f)
    uuid = name.replace('.wav', '')
    row: dict = meta[uuid]
    data.append((uuid, dict(path=f, **row)))
  # === 3. shuffle and split
  rand = np.random.RandomState(seed=seed)
  rand.shuffle(data)
  n = len(data)
  start = int(n * start)
  end = int(n * end)
  data = dict(data[start:end])
  # === 4. save to JSON
  path = os.path.join(CACHE_PATH,
                      f'{partition}_{start:g}_{end:g}_{seed:d}.json')
  with open(path, 'w') as f:
    json.dump(data, f)
  return Path(path)


