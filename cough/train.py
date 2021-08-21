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
import glob
import json
import os
import random
import re
import shutil
import zipfile
from collections import namedtuple
from typing import Union, NamedTuple, Tuple, Dict, List, Sequence, Optional, Any

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

from config import SEED, META_DATA, get_json, SR, SAVE_PATH, WAV_META
from features import AcousticFeatures, AudioRead, VAD, LabelEncoder
from models import *
from utils import save_allfig
import pytorch_lightning as pl
from torch import nn
from logging import getLogger

logger = getLogger(__name__)

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
    outputs: Sequence[str] = ('signal', 'result'),
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
class TerminateOnNaN(pl.callbacks.Callback):

  def on_train_epoch_end(self, trainer, pl_module, unused=None):
    metr = trainer.callback_metrics
    for k, v in metr.items():
      if torch.any(torch.isnan(v)):
        logger.info(f'Terminate On NaNs {k}={v}')
        trainer.should_stop = True


class TrainModule(pl.LightningModule):

  def __init__(self, model, lr: float = 1e-4):
    super().__init__()
    self.model = model
    self.lr = lr
    self.fn_loss = torch.nn.BCEWithLogitsLoss()

  def forward(self, x: PaddedBatch, proba: bool = False):
    # in lightning, forward defines the prediction/inference actions
    y = self.model(x)
    if proba:
      y = torch.sigmoid(y)
    return y

  def predict_step(self,
                   batch: Any,
                   batch_idx: int,
                   dataloader_idx: Optional[int] = None) -> Any:
    print(batch)
    exit()

  def on_train_batch_start(self,
                           batch: Any,
                           batch_idx: int,
                           dataloader_idx: int):
    pass

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
    self.log('valid_acc', acc, prog_bar=True)
    return dict(acc=acc)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    return optimizer


def get_model_path(model, overwrite: bool = False) -> Tuple[str, str]:
  path = os.path.join(SAVE_PATH, model.name)
  if overwrite and os.path.exists(path):
    print('Overwrite path:', path)
    shutil.rmtree(path)
  if not os.path.exists(path):
    os.makedirs(path)
  print('Save model at path:', path)
  # higher is better
  best_path = sorted(
    glob.glob(f'{path}/**/model-*.ckpt', recursive=True),
    key=lambda p: float(
      next(re.finditer(r'valid_acc=\d+\.\d+', p)).group().split('=')[-1]),
    reverse=True)
  if len(best_path) > 0:
    best_path = best_path[0]
  else:
    best_path = None
  print('Best model at path:', best_path)
  return path, best_path


def train_model(args: argparse.Namespace,
                model: torch.nn.Module,
                train: DynamicItemDataset,
                valid: Optional[DynamicItemDataset] = None,
                epochs: int = 100,
                n_cpu: int = 4):
  path, best_path = get_model_path(model, overwrite=args.overwrite)

  model = TrainModule(model)
  trainer = pl.Trainer(
    gpus=1,
    default_root_dir=path,
    callbacks=[
      pl.callbacks.ModelCheckpoint(filename='model-{valid_acc:.2f}',
                                   monitor='valid_acc',
                                   mode='max',
                                   save_top_k=5,
                                   verbose=True),
      pl.callbacks.EarlyStopping('valid_acc',
                                 mode='max',
                                 patience=10,
                                 verbose=True),
      TerminateOnNaN(),
    ],
    max_epochs=epochs,
    val_check_interval=100,
    resume_from_checkpoint=best_path,
  )

  trainer.fit(model,
              train_dataloaders=to_loader(train, n_cpu),
              val_dataloaders=None if valid is None else to_loader(valid, 0))


def evaluate(model: torch.nn.Module):
  path, best_path = get_model_path(model)
  if best_path is None:
    raise RuntimeError(f'No model found at path: {path}')
  model = TrainModule.load_from_checkpoint(
    checkpoint_path=best_path,
    model=model)
  # the pretrained model cannot be switch to CPU easily
  # model.cpu()
  model.eval()

  test_key = ['final_pub_test']
  with torch.no_grad():
    for key in test_key:
      test = init_dataset(key, outputs=('signal', 'id'))
      results = dict()
      for batch in tqdm(to_loader(test, 0), desc=key):
        y_proba = model(batch, proba=True).cpu().numpy()
        for k, v in zip(batch.id, y_proba):
          results[k] = v
      uuid_order = list(META_DATA['final_pub_test'].values())[0].uuid
      text = 'uuid,assessment_result\n'
      for k in uuid_order:
        text += f'{k},{results[k]}\n'
      csv_path = os.path.join(path, f'{key}.csv')
      with open(csv_path, 'w') as f:
        f.write(text[:-1])
      print('Save results to:', csv_path)


# ===========================================================================
# For predicting
# ===========================================================================
def main(args: argparse.Namespace):
  train = init_dataset('final_train', split=(0., 0.8))
  valid = init_dataset('final_train', split=(0.8, 1.0))
  features = [pretrained_xvec()]
  model = SimpleClassifier(features, name='pre_xvec')
  if args.eval:
    evaluate(model)
  else:
    train_model(args, model, train, valid=valid)


# ===========================================================================
# Main
# ===========================================================================
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--eval', action='store_true')
  parsed_args = parser.parse_args()
  main(parsed_args)
