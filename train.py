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
import os
import random
import shutil
from collections import namedtuple
from typing import Union, NamedTuple

import numpy as np
import pandas as pd
import seaborn as sns
import speechbrain
import torch
from matplotlib import pyplot as plt
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import ReproducibleRandomSampler
from speechbrain.nnet.losses import bce_loss
from speechbrain.utils.checkpoints import Checkpointer
from speechbrain.utils.epoch_loop import EpochCounter
from tqdm import tqdm
from typing_extensions import Literal

from const import SEED, META_DATA
from utils import save_allfig

# ===========================================================================
# Load data
# ===========================================================================
np.random.seed(SEED)
torch.random.manual_seed(SEED)
random.seed(SEED)


def data_exploration():
  train: pd.DataFrame = META_DATA['train'].copy(deep=True)
  test0: pd.DataFrame = META_DATA['pub_test0'].copy(deep=True)
  test1: pd.DataFrame = META_DATA['pri_test0'].copy(deep=True)
  g = sns.PairGrid(train, hue="subject_gender")
  g.map_diag(sns.histplot)
  g.map_offdiag(sns.scatterplot)
  g.add_legend()
  plt.savefig('/tmp/tmp.pdf')


Batch = namedtuple('Batch', ['specs', 'results', 'lengths'])


def collate(data) -> NamedTuple:
  specs = []
  results = []
  lengths = []
  for x in data:
    if x['spec'] is None:
      continue
    specs.append(x['spec'].unsqueeze(0))
    results.append(x['result'])
    lengths.append(x['length'])
  specs = torch.cat(specs, 0)
  results = torch.tensor(results, dtype=specs.dtype)
  lengths = torch.clamp(torch.tensor(lengths, dtype=specs.dtype), max=1.0)
  return Batch(specs, results, lengths)


def create_dataset(
    json_path,
    batch_size: int = 16,
    num_workers: int = 0,
    mode: Literal['train', 'eval', 'check'] = 'train'
) -> Union[DynamicItemDataset, SaveableDataLoader]:
  feat = AcousticFeatures(training=mode != 'eval')
  ds = DynamicItemDataset.from_json(json_path)
  ds.add_dynamic_item(
    feat,
    takes=['path'],
    provides=['signal', 'energies', 'spec', 'vad', 'length', 'status'])
  ds = ds.filtered_sorted(key_test=dict(status=lambda val: val))
  if mode == 'check':
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
  elif mode == 'train':
    ds.set_output_keys(['path', 'spec', 'result', 'length'])
  else:
    ds.set_output_keys(['path', 'spec', 'result', 'length'])
  # create data loader
  ds = SaveableDataLoader(
    dataset=ds,
    sampler=ReproducibleRandomSampler(ds),
    collate_fn=collate,
    num_workers=num_workers,
    drop_last=False,
    batch_size=batch_size,
    pin_memory=True if torch.cuda.is_available() else False,
  )
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
  train = create_dataset(JSON_TRAIN, num_workers=4)
  valid = create_dataset(JSON_VALID, num_workers=2, mode='eval')

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
  parser.add_argument('--overwrite', action='store_true')
  parsed_args = parser.parse_args()
  main(parsed_args)
