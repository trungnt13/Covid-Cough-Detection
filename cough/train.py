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
from typing import Union, NamedTuple, Tuple, Dict, List, Sequence

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

from config import SEED, META_DATA, get_json, SR, SAVE_PATH
from features import AcousticFeatures, AudioRead, VAD, LabelEncoder
from models import *
from utils import save_allfig
import pytorch_lightning as pl
from torch import nn

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


def init_dataset(
    partition: Union[str, Sequence[str]],
    split: Tuple[float, float] = (0.0, 1.0),
    random_cut: float = 3,
    outputs: List[str] = ('signal', 'result'),
) -> Union[DynamicItemDataset, SaveableDataLoader]:
  json_path = get_json(partition, start=split[0], end=split[1])
  # path, meta, id
  ds = DynamicItemDataset.from_json(json_path)
  # signal, sr, meta
  ds.add_dynamic_item(AudioRead(sr=SR, random_cut=random_cut),
                      takes=AudioRead.takes,
                      provides=AudioRead.provides)
  # vad, energies
  ds.add_dynamic_item(VAD(sr=SR), takes=VAD.takes, provides=VAD.provides)
  # result, gender, age
  ds.add_dynamic_item(LabelEncoder(), takes=LabelEncoder.takes,
                      provides=LabelEncoder.provides)
  ds.set_output_keys(outputs)
  return ds


def to_loader(ds: DynamicItemDataset,
              num_workers: int = 0,
              batch_size: int = 16):
  # create data loader
  return SaveableDataLoader(
    dataset=ds,
    sampler=ReproducibleRandomSampler(ds, seed=SEED),
    collate_fn=PaddedBatch,
    num_workers=num_workers,
    drop_last=False,
    batch_size=batch_size,
    pin_memory=True if torch.cuda.is_available() else False,
  )


# ds = create_dataset(JSON_TRAIN, mode='train')
# for x in tqdm(ds):
#   pass
# exit()

# ===========================================================================
# For training
# ===========================================================================
class TrainModule(pl.LightningModule):

  def __init__(self, model, lr: float = 1e-4):
    super().__init__()
    self.model = model
    self.lr = lr
    self.fn_loss = torch.nn.BCEWithLogitsLoss()

  def forward(self, x):
    # in lightning, forward defines the prediction/inference actions
    return self.model(x)

  def training_step(self, batch, batch_idx):
    y_pred = self.model(batch).float()
    y_true = batch.result.data.float().cuda()
    loss = self.fn_loss(y_pred, y_true)
    # Logging to TensorBoard by default
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    y_pred = torch.sigmoid(self.model(batch)).ge(0.5)
    y_true = batch.result.data.float().cuda()
    acc = y_pred.eq(y_true).float().sum() / y_pred.shape[0]
    return acc

  def validation_epoch_end(self, outputs):
    acc = np.mean([o.cpu().numpy() for o in outputs])
    self.log('valid_acc', acc)
    print('Accuracy:', acc)
    return dict(acc=acc)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer


def main(args: argparse.Namespace):
  train = init_dataset('final_train', split=(0, 0.8))
  valid = init_dataset('final_train', split=(0.8, 1.0))
  features = [pretrained_xvec(), pretrained_ecapa(), pretrained_langid()]
  model = TrainModule(SimpleClassifier(features))
  trainer = pl.Trainer(
    gpus=1,
    default_root_dir=SAVE_PATH,
    terminate_on_nan=True,
    callbacks=[
      pl.callbacks.ModelCheckpoint(verbose=True,
                                   save_on_train_epoch_end=True),
    ])
  trainer.fit(model,
              train_dataloaders=to_loader(train, 4),
              val_dataloaders=to_loader(valid, 0),
              )


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--overwrite', action='store_true')
  parsed_args = parser.parse_args()
  main(parsed_args)
