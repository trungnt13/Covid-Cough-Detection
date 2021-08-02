"""
conda create -n covid -c conda-forge  python=3.7
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas pyro-ppl speechbrain matplotlib seaborn librosa

Outlier:
- aicv115m_public_train/train_audio_files_8k/7f186559-2e13-4d76-bedb-2409363abf1d.wav
- aicv115m_public_train/train_audio_files_8k/81bc3080-b30b-403f-80e2-47396677d9f6.wav
- aicv115m_public_train/train_audio_files_8k/bb7a3b30-88d9-4101-a39d-a4bac9a436b8.wav
- aicv115m_public_train/train_audio_files_8k/2ce77921-b508-43fd-a8ee-c0b09f28901c.wav

https://colab.research.google.com/drive/1JJc4tBhHNXRSDM2xbQ3Z0jdDQUw4S5lr?usp=sharing

VAD:
- https://kaldi-asr.org/doc/voice-activity-detection_8cc_source.html
- https://github.com/pytorch/audio/blob/master/examples/interactive_asr/vad.py#L38
"""
import argparse
import json
import os
import random
import shutil
from collections import namedtuple
from typing import Union, NamedTuple, Tuple, Dict, List

import numpy as np
import pandas as pd
import seaborn
import seaborn as sns
import speechbrain
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
from six import string_types
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from speechbrain.nnet.losses import bce_loss
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.epoch_loop import EpochCounter
from tqdm import tqdm
from typing_extensions import Literal

from const import SEED, META_DATA, get_json, SR
from features import AcousticFeatures, AudioRead, VAD, Labeler
from utils import save_allfig

# ===========================================================================
# Load data
# 'uuid'
# 'subject_age'
# 'subject_gender'
# 'audio_noise_note'
# 'cough_intervals'
# 'assessment_result'
#####
# 'symptoms_status_choice'
# 'medical_condition_choice'
# 'smoke_status_choice'
# 'insomnia_status_choice'
####
# 'duration'
# 'sr'
# ===========================================================================
np.random.seed(SEED)
torch.random.manual_seed(SEED)
random.seed(SEED)


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
    seaborn.histplot(arr, ax=plt.subplot(1, len(vars), idx + 1))
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
  exit()


# data_exploration()


def create_dataset(
    partition: str,
    split: Tuple[float, float] = (0.0, 1.0),
    batch_size: int = 16,
    num_workers: int = 0,
    debug: bool = False
) -> Union[DynamicItemDataset, SaveableDataLoader]:
  json_path = get_json(partition, start=split[0], end=split[1])
  is_training = 'train' in partition
  ds = DynamicItemDataset.from_json(json_path)
  ds.add_dynamic_item(AudioRead(sr=SR, random_cut=3),
                      takes=AudioRead.takes,
                      provides=AudioRead.provides)
  ds.add_dynamic_item(VAD(sr=SR), takes=VAD.takes, provides=VAD.provides)
  ds.add_dynamic_item(Labeler(), takes=Labeler.takes, provides=Labeler.provides)
  ds.set_output_keys(['signal', 'result'])

  # ds = ds.filtered_sorted(key_test=dict(status=lambda val: val))

  def _plot_wave():
    ds.set_output_keys(['signal', 'meta', 'sr'])
    plt.figure(figsize=(10, 20))
    for i, j in enumerate(np.random.choice(len(ds), 10)):
      ax = plt.subplot(10, 1, i + 1)
      x = ds[int(j)]
      ax.plot(x['signal'].numpy())
      cough = x['meta']['cough_intervals']
      if isinstance(cough, list):
        for y in cough:
          start = int(y['start'] * x['signal'].shape[0])
          end = int(y['end'] * x['signal'].shape[0])
          ax.add_patch(mpl.patches.Rectangle((start, -1), end - start, 2,
                                             fill=True,
                                             color="red",
                                             alpha=0.3,
                                             linewidth=1))
      plt.title(
        f"{x['meta']['assessment_result']}-{x['meta']['cough_intervals']}",
        fontsize=6)
    plt.tight_layout()

  def _plot_check():
    ds.set_output_keys(['path', 'signal', 'energies', 'vad',
                        'spec', 'gender', 'age', 'result'])
    for i in tqdm(list(range(len(ds)))):
      if random.random() < 0.1:
        x = ds[i]
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(x['signal'])
        plt.title(f"{x['path']} - {x['gender']}:{x['age']}:{x['result']}",
                  fontsize=6)
        plt.axis('off')
        plt.subplot(4, 1, 2)
        plt.plot(x['energies'])
        plt.axis('off')
        plt.subplot(4, 1, 3)
        vad = x['vad'].numpy()
        plt.scatter(np.arange(len(vad)), vad, s=8)
        plt.subplot(4, 1, 4)
        plt.imshow(x['spec'].numpy().T, aspect='auto', origin='lower')
        plt.axis('off')
        plt.tight_layout()
    save_allfig()
    return ds

  # create data loader
  ds = SaveableDataLoader(
    dataset=ds,
    sampler=ReproducibleRandomSampler(ds, seed=SEED),
    collate_fn=PaddedBatch,
    num_workers=num_workers,
    drop_last=False,
    batch_size=batch_size,
    pin_memory=True if torch.cuda.is_available() else False,
  )
  for data in ds:
    data: PaddedBatch
    print(data['signal'].data.shape)
    print(data['result'].shape)
    exit()
  exit()
  return ds


# ds = create_dataset(JSON_TRAIN, mode='train')
# for x in tqdm(ds):
#   pass
# exit()

# ===========================================================================
# Create network
# ===========================================================================
def create_network():
  pass


def main(args: argparse.Namespace):
  train = create_dataset('final_train', split=(0, 0.8), num_workers=0)
  valid = create_dataset('final_train', split=(0.8, 1.0), num_workers=0)

  model_path = os.path.join(SAVE_PATH, 'model_test')
  if args.overwrite and os.path.exists(model_path):
    print('Overwrite:', model_path)
    shutil.rmtree(model_path)

  class CovidBrain(speechbrain.Brain):

    def compute_forward(self, data: Batch, stage):
      x = data.specs.to(self.device)
      x = self.modules.encoder(x, lens=data.lengths)
      x = x.squeeze(1)
      x = self.modules.output(x)
      return x

    def compute_objectives(self, logits, batch, stage):
      y_true = batch.results.to(self.device)
      loss = bce_loss(logits, y_true)
      # if stage == speechbrain.Stage.TRAIN:
      #   loss = bce_loss(logits, y_true)
      # else:
      #   x = self.compute_forward(batch, stage)
      #   x = torch.sigmoid(x).ge(0.5).squeeze(1).type(y_true.dtype)
      #   loss = torch.eq(x, y_true).sum() / y_true.shape[0]
      return loss

    def on_stage_end(self, stage, stage_loss, epoch=None):
      if stage == speechbrain.Stage.VALID:
        print(f'Valid: {stage_loss:.4f}')
        # ckp = self.checkpointer.find_checkpoint(max_key='valid_acc')
        # if ckp is None or ckp.meta['valid_acc'] <= stage_loss:
        if np.isnan(stage_loss):
          print('Recover checkpoint')
          self.checkpointer.recover_if_possible()
        else:
          print('Save checkpoint')
          self.checkpointer.save_and_keep_only(meta=dict(valid=stage_loss),
                                               keep_recent=True,
                                               num_to_keep=5)

  from speechbrain.lobes.models.Xvector import Xvector
  modules = {"encoder": Xvector(device='cuda',
                                in_channels=80,
                                lin_neurons=512),
             "output": torch.nn.Sequential(torch.nn.Dropout(p=0.7),
                                           torch.nn.Linear(512, 512),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.BatchNorm1d(512),
                                           torch.nn.Linear(512, 512),
                                           torch.nn.LeakyReLU(),
                                           torch.nn.BatchNorm1d(512),
                                           torch.nn.Linear(512, 1)),
             "sigmoid": torch.nn.Sigmoid()}
  brain = CovidBrain(
    modules,
    run_opts=dict(device='cuda:0',
                  ckpt_interval_minutes=-1,
                  noprogressbar=False),
    opt_class=lambda params: torch.optim.Adam(params, 1e-4),
    checkpointer=Checkpointer(model_path))
  epoch = EpochCounter(10000)
  brain.fit(epoch,
            train_set=train,
            valid_set=valid)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--overwrite', action='store_true')
  parsed_args = parser.parse_args()
  main(parsed_args)
