from typing import List, Dict, Union

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from six import string_types

from config import META_DATA, WAV_META
from utils import save_allfig

sns.set()


def medical_condition():
  df: pd.DataFrame = META_DATA['final_train']['public_train_medical_condition']
  n = len(df.columns) - 1  # no uuid
  n_col = 2
  n_row = int(np.ceil(n / 3))
  plt.figure(figsize=(4 * n_col, 4 * n_row))
  count = 0
  for i, name in enumerate(df.columns):
    if name == 'uuid':
      continue
    count += 1
    val: pd.Series = df[name]
    print(f'Plotting {name}', len(val.unique()))
    ax = plt.subplot(n_row, n_col, count)
    sns.histplot(val, ax=ax)
    plt.xticks(rotation=90, fontsize=6)
    plt.title(name)
  plt.tight_layout()
  save_allfig('/tmp/tmp.pdf')



def training_duration():
  for k, v in WAV_META.items():
    duration = [i[0] for i in v.values()]
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    sns.histplot(duration, bins=50, kde=True)
    plt.title(k)
    plt.subplot(1, 2, 2)
    sns.histplot(duration, bins=50, cumulative=True, stat='density')
    plt.tight_layout()
  save_allfig('/tmp/tmp.pdf')


def data_exploration():
  ##
  vars = ['symptoms_status_choice', 'medical_condition_choice',
          'smoke_status_choice', 'insomnia_status_choice']
  df: pd.DataFrame = META_DATA['final_train']['public_train_medical_condition']
  plt.figure(figsize=(len(vars) * 3, 4))
  for idx, v in enumerate(vars):
    arr = []
    for i in df[v]:
      if '[' in i:
        i = eval(i)
        arr += i
      else:
        arr.append(i)
    sns.histplot(arr, ax=plt.subplot(1, len(vars), idx + 1))
    plt.xticks(rotation=90)
    plt.title(v)
  plt.tight_layout()
  ##
  vars = ['subject_age', 'subject_gender', 'assessment_result']
  name, df = list(META_DATA['train'].items())[0]
  df: pd.DataFrame = df.copy(deep=True)
  plt.figure(figsize=(5 * len(vars), 3))
  for i, v in enumerate(vars):
    sns.histplot(df[v].apply(str), ax=plt.subplot(1, 3, i + 1))
    plt.xticks(rotation=90)
  plt.suptitle(name)
  ##
  name, df = list(META_DATA['final_train'].items())[0]
  df: pd.DataFrame = df.copy(deep=True)
  plt.figure(figsize=(5 * len(vars), 3))
  for i, v in enumerate(vars):
    sns.histplot(df[v].apply(str), ax=plt.subplot(1, 3, i + 1))
    plt.xticks(rotation=90)
  plt.suptitle(name)
  ##
  plt.figure(figsize=(12, 3))
  s: pd.Series = df.audio_noise_note.apply(str).value_counts()
  n = 30
  plt.bar(np.arange(0, n), s.values[:n])
  plt.xticks(np.arange(0, n), s.keys()[:n], rotation=90)
  ##
  data: pd.Series = df.cough_intervals
  is_nan = np.array([False if isinstance(i, string_types) else (i is np.nan)
                     for i in data.values])
  plt.figure()
  sns.histplot(df.assessment_result[is_nan],
               ax=plt.subplot(1, 2, 1))
  plt.title('is_nan=True')
  sns.histplot(df.assessment_result[np.logical_not(is_nan)],
               ax=plt.subplot(1, 2, 2))
  plt.title('is_nan=False')
  plt.suptitle(f'Cough Interval {np.sum(is_nan)}/{len(is_nan)}')
  plt.tight_layout()
  events: List[List[Dict[str, Union[float, List]]]] = [
    eval(row) for row in data if row is not np.nan]
  labels = sum([[i['labels'] for i in e] for e in events], [])
  print('Labels:', np.unique(labels))
  duration = [[i['end'] - i['start'] for i in e] for e in events]
  plt.figure(figsize=(15, 4))
  sns.histplot(sum(duration, []), bins=50, ax=plt.subplot(1, 3, 1))
  plt.title('Universal cough duration')
  sns.histplot([np.sum(i) for i in duration], bins=50, ax=plt.subplot(1, 3, 2))
  plt.title('Per utterance cough duration')
  sns.histplot([len(i) for i in duration], ax=plt.subplot(1, 3, 3))
  plt.title('#cough per utterance')
  plt.tight_layout()
  ##
  save_allfig('/tmp/tmp.pdf')
