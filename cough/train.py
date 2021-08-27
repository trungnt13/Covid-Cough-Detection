"""
conda create -f covid.yml
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
import json
import argparse
import glob
import inspect
import pickle
import random
import re
import shutil
import zipfile
from collections import defaultdict
from typing import Tuple
import traceback

import numpy as np
import pytorch_lightning as pl
import torch.nn
from six import string_types
from sklearn.metrics import auc as auc_score, roc_curve, confusion_matrix, \
  f1_score, accuracy_score
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler, \
  ReproducibleRandomSampler
from torch.optim import lr_scheduler
from tqdm import tqdm

from config import META_DATA, get_json, SAVE_PATH, COVI_WEIGHT, ZIP_FILES, \
  DATA_SEED, PSEUDOLABEL_PATH, GEN_WEIGHT, PSEUDO_AGEGEN, write_errors
from features import AudioRead, VAD, LabelEncoder, PseudoLabeler, Pitch, MixUp
from models import *

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
pl.seed_everything(SEED)
CFG = Config()


@dataclasses.dataclass()
class Partition:
  name: str = 'final_train'
  start: float = 0.0
  end: float = 1.0


def init_dataset(
    partition: Union[Partition, List[Partition]],
    random_cut: float = -1,
    is_training: bool = True,
    outputs: Sequence[str] = ('signal', 'result'),
    only_result: Optional[int] = None
) -> Union[DynamicItemDataset, SaveableDataLoader]:
  # prepare json file
  if isinstance(partition, Partition):
    json_path = get_json(partition.name,
                         start=partition.start, end=partition.end)
  else:
    json_path = [get_json(p.name, start=p.start, end=p.end)
                 for p in partition]
    name = '_'.join([os.path.basename(i).split('.')[0] for i in json_path])
    name += '.json'
    data = dict()
    for path in json_path:
      with open(path, 'r') as f:
        data.update(json.load(f))
    data = [(k, v) for k, v in data.items()]
    np.random.shuffle(data)
    data = dict(data)
    json_path = os.path.join(CACHE_PATH, name)
    with open(json_path, 'w') as f:
      json.dump(data, f)
  # path, meta, id
  ds = DynamicItemDataset.from_json(json_path)
  # pre-filter much faster
  if only_result is not None:
    for idx in list(ds.data_ids):
      result = ds.data[idx]['meta']['assessment_result']
      if CFG.pseudolabel:
        labeler = PseudoLabeler.get_labeler(pseudo_soft=CFG.pseudosoft,
                                            pseudo_rand=CFG.pseudorand)
      if result == 'unknown':
        result = labeler.label(ds.data[idx]['meta']['uuid']) \
          if CFG.pseudolabel else -1
      if result != only_result:
        ds.data_ids.remove(idx)
        del ds.data[idx]
  # signal, sr, meta
  ds.add_dynamic_item(AudioRead(random_cut=random_cut),
                      takes=AudioRead.takes,
                      provides=AudioRead.provides)
  ds.add_dynamic_item(Pitch(),
                      takes=Pitch.takes,
                      provides=Pitch.provides)
  # vad, energies
  ds.add_dynamic_item(VAD(), takes=VAD.takes, provides=VAD.provides)
  # result, gender, age
  ds.add_dynamic_item(LabelEncoder(pseudo_labeling=CFG.pseudolabel,
                                   pseudo_rand=CFG.pseudorand,
                                   pseudo_soft=CFG.pseudosoft),
                      takes=LabelEncoder.takes,
                      provides=LabelEncoder.provides)
  # DO NOT mixing for evaluation
  if CFG.mixup and is_training:
    print(' * Enable MixUp:', partition)
    ds.add_dynamic_item(MixUp(contrastive=CFG.task == 'contrastive'),
                        takes=MixUp.takes,
                        provides=MixUp.provides)
  ds.set_output_keys(outputs)
  return ds


def to_loader(ds: DynamicItemDataset,
              num_workers: int = 0,
              is_training=True,
              drop_last=False,
              batch_size=8):
  sampler = None
  if is_training:
    if CFG.oversampling:
      if CFG.task == 'gender':
        keys = ['subject_gender', 'subject_age']
      elif CFG.task == 'covid':
        keys = ['assessment_result']
      else:
        raise NotImplementedError(f'No support for task={CFG.task}.')
      # oversampling with replacement
      weights = {}
      counts = defaultdict(int)
      for v in ds.data.values():
        meta = v['meta']
        uuid = meta['uuid']
        k = '.'.join([str(meta.get(i, 'unknown')) for i in keys])
        weights[uuid] = k
        counts[k] += 1
      # normalize
      total = sum(i for i in counts.values())
      counts = {k: total / v
                for k, v in counts.items()}
      # weight for each example
      weights = [counts[weights[idx]] for idx in ds.data_ids]
      sampler = ReproducibleWeightedRandomSampler(weights,
                                                  num_samples=len(ds),
                                                  replacement=True,
                                                  seed=SEED)
    else:
      sampler = ReproducibleRandomSampler(ds, seed=SEED)
  # create data loader
  bs = CFG.bs if is_training else 8
  if batch_size is not None:
    bs = batch_size
  return SaveableDataLoader(
    dataset=ds,
    sampler=sampler,
    collate_fn=PaddedBatch,
    num_workers=num_workers,
    drop_last=drop_last,
    batch_size=bs,
    pin_memory=True if torch.cuda.is_available() else False,
  )


def get_model_path(model, overwrite=False, monitor='val_f1') -> Tuple[str, str]:
  overwrite = CFG.overwrite & overwrite
  prefix = '' if len(CFG.prefix) == 0 else f'{CFG.prefix}_'
  path = os.path.join(SAVE_PATH, f'{prefix}{model.name}_{DATA_SEED}')
  if overwrite and os.path.exists(path):
    print(' * Overwrite path:', path)
    backup_path = path + '.bk'
    if os.path.exists(backup_path):
      shutil.rmtree(backup_path)
    print(' * Backup path:', backup_path)
    shutil.copytree(path, backup_path)
    shutil.rmtree(path)
  if not os.path.exists(path):
    os.makedirs(path)
  print(' * Save model at path:', path)
  # higher is better
  checkpoints = glob.glob(f'{path}/**/model-*.ckpt', recursive=True)
  pattern = f'{monitor}=\d+\.\d+'
  checkpoints = list(filter(lambda s: len(re.findall(pattern, s)) > 0,
                            checkpoints))
  if len(checkpoints) > 0:
    if 'loss' in monitor:
      reverse = False  # smaller better
    else:
      reverse = True  # higher better
    best_path = sorted(
      checkpoints,
      key=lambda p: float(
        next(re.finditer(pattern, p)).group().split('=')[-1]),
      reverse=reverse)
    best_path = best_path[CFG.top]
  else:
    best_path = None
  print(' * Best model at path:', best_path)
  return path, best_path


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

  def __init__(self,
               model: CoughModel,
               target: Union[str, List[str]] = 'result'):
    super().__init__()
    self.model = model
    if isinstance(target, string_types):
      target = [target]
    target2weight = dict(
      gender=GEN_WEIGHT,
      age=AGE_WEIGHT,
      result=COVI_WEIGHT,
      covid=COVI_WEIGHT
    )
    weights = [target2weight.get(i, torch.tensor(1. / CFG.pos_weight_rescale))
               for i in target]
    self.fn_bce = [
      torch.nn.BCEWithLogitsLoss(pos_weight=CFG.pos_weight_rescale * w)
      for w in weights]
    self.target = target
    self.label_noise = float(CFG.label_noise)
    self.lr = float(CFG.lr)

  def forward(self, x: PaddedBatch, proba: bool = False):
    # in lightning, forward defines the prediction/inference actions
    y = self.model(x)
    if proba:
      if isinstance(y, (tuple, list)):
        y = [torch.sigmoid(i) for i in y]
      else:
        y = torch.sigmoid(y)
    return y

  def training_step(self, batch, batch_idx):
    losses = 0.
    extra_losses = 0.
    y_pred = self.model(batch)
    if not isinstance(y_pred, (tuple, list)):
      y_pred = [y_pred]
    for i, (pred, target, fn_loss) in enumerate(zip(
        y_pred, self.target, self.fn_bce)):
      if isinstance(pred, ModelOutput):
        extra_losses += pred.losses
        pred = pred.outputs
      pred = pred.float()
      true = getattr(batch, target).data
      if torch.cuda.is_available():
        true.cuda()
      true = true.float()
      if self.label_noise is not None and self.label_noise > 0:
        z = torch.rand(true.shape, device=true.get_device()) * self.label_noise
        true = true * (1 - z) + (1 - true) * z
      losses += fn_loss(pred, true)
    # Logging to TensorBoard by default
    losses += extra_losses
    self.log("train_loss", losses)
    return losses

  def validation_step(self, batch, batch_idx):
    y_true = []
    y_pred = []
    pred = self.model(batch)
    if not isinstance(pred, (tuple, list)):
      pred = [pred]
    for pred, target in zip(pred, self.target):
      if isinstance(pred, ModelOutput):
        pred = pred.outputs
      if len(pred.shape) == 1:
        pred = torch.sigmoid(pred).ge(0.5)
      else:
        pred = torch.argmax(pred, -1)
      true = getattr(batch, target).data.float().cuda()
      y_pred.append(pred)
      y_true.append(true)
    return dict(true=y_true, pred=y_pred)

  def validation_epoch_end(self, outputs):
    all_auc = []
    all_f1 = []
    all_acc = []
    print('\n\n')
    for i, target in enumerate(self.target):
      # calculate AUC
      true = torch.cat([o['true'][i] for o in outputs], 0).cpu().numpy()
      pred = torch.cat([o['pred'][i] for o in outputs], 0).cpu().numpy()
      # f1 score
      f1 = f1_score(true, pred)
      if np.isnan(f1) or np.isinf(f1):
        f1 = 0.
      # row: true_labels
      print(f'{target}:\n{confusion_matrix(true, pred)}\n')
      # AUC
      fpr, tpr, thresholds = roc_curve(true, pred, pos_label=1)
      auc = auc_score(fpr, tpr)
      if np.isnan(auc) or np.isinf(auc):
        auc = 0
      # ACC
      acc = accuracy_score(true, pred)
      if len(self.target) > 1:
        self.log(f'{target[:3]}_acc', torch.tensor(acc), prog_bar=False)
        self.log(f'{target[:3]}_auc', torch.tensor(auc), prog_bar=False)
        self.log(f'{target[:3]}_f1', torch.tensor(f1), prog_bar=True)
      all_acc.append(acc)
      all_f1.append(f1)
      all_auc.append(auc)
    # aggregate all scores
    acc = np.mean(all_acc)
    auc = np.mean(all_auc)
    f1 = np.mean(all_f1)
    self.log('val_acc', torch.tensor(acc), prog_bar=True)
    self.log('val_auc', torch.tensor(auc), prog_bar=True)
    self.log('val_f1', torch.tensor(f1), prog_bar=True)
    # print the learning rate
    for pg in self.optimizers().param_groups:
      self.log('lr', pg.get('lr'), prog_bar=True)
    return dict(acc=acc, auc=auc, f1=f1)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
    if len(CFG.scheduler) > 0:
      if CFG.scheduler == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=CFG.gamma)
      elif CFG.scheduler == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, gamma=CFG.gamma,
                                        step_size=CFG.lr_step)
      elif CFG.scheduler == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.lr_step,
                                                   eta_min=1e-10)
      elif CFG.scheduler == 'cyc':
        scheduler = lr_scheduler.CyclicLR(optimizer,
                                          base_lr=CFG.lr / 10,
                                          max_lr=CFG.lr * 10,
                                          step_size_up=CFG.lr_step)
      else:
        raise NotImplementedError(f'Unknown LR scheduler {CFG.scheduler}')
      return [optimizer], [scheduler]
    return optimizer


def train_covid_detector(model: CoughModel,
                         train: DynamicItemDataset,
                         valid: Optional[DynamicItemDataset] = None,
                         target: Union[str, List[str]] = 'result'):
  monitor = CFG.monitor
  path, best_path = get_model_path(
    model, overwrite=False,
    monitor=CFG.load if len(CFG.load) > 0 else monitor)

  model = TrainModule(model, target=target)
  trainer = pl.Trainer(
    gpus=1,
    default_root_dir=path,
    gradient_clip_val=CFG.grad_clip,
    gradient_clip_algorithm='norm',
    callbacks=[
      pl.callbacks.ModelCheckpoint(filename='model-{%s:.2f}' % monitor,
                                   monitor=monitor,
                                   mode='max',
                                   save_top_k=20,
                                   verbose=True),
      pl.callbacks.ModelCheckpoint(save_last=True,
                                   verbose=True),
      pl.callbacks.EarlyStopping(monitor,
                                 mode='max',
                                 patience=CFG.patience,
                                 verbose=True),
      TerminateOnNaN(),
    ],
    max_epochs=CFG.epochs,
    val_check_interval=0.8,
    resume_from_checkpoint=best_path,
  )
  # int(300 / (CFG.bs / 16))
  trainer.fit(
    model,
    train_dataloaders=to_loader(train, num_workers=CFG.ncpu),
    val_dataloaders=None if valid is None else
    to_loader(valid, num_workers=2, is_training=False))


# ===========================================================================
# Training contrastive model
# ===========================================================================
class ContrastiveModule(TrainModule):

  def forward(self, X: PaddedBatch):
    return self.model(X)

  def training_step(self, batch, batch_idx):
    anchor, pos, neg = batch['a'], batch['p'], batch['n']
    loss = self.model([anchor, pos, neg])
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    anchor, pos, neg = batch['a'], batch['p'], batch['n']
    loss = self.model([anchor, pos, neg], reduce=False)
    return dict(val_loss=loss)

  def validation_epoch_end(self, outputs):
    val_loss = torch.cat([i['val_loss'] for i in outputs]).mean()
    self.log('val_loss', val_loss, prog_bar=True)
    return dict(val_loss=val_loss)


def train_contrastive(model: CoughModel,
                      train: List[DynamicItemDataset],
                      valid: List[DynamicItemDataset]):
  path, best_path = get_model_path(model, overwrite=False, monitor='val_loss')
  # no need oversampling
  CFG.oversampling = False

  train = [to_loader(i, num_workers=max(1, CFG.ncpu // 2),
                     is_training=True, drop_last=True)
           for i in train]
  valid = [to_loader(i, num_workers=2, is_training=False, drop_last=True,
                     batch_size=5)
           for i in valid]

  model = ContrastiveModule(model)
  trainer = pl.Trainer(
    gpus=1,
    default_root_dir=path,
    gradient_clip_val=CFG.grad_clip,
    gradient_clip_algorithm='norm',
    callbacks=[
      pl.callbacks.ModelCheckpoint(filename='model-{val_loss:.2f}',
                                   monitor='val_loss',
                                   mode='min',
                                   save_last=True,
                                   save_top_k=20,
                                   verbose=True),
      pl.callbacks.EarlyStopping('val_loss',
                                 mode='min',
                                 patience=CFG.patience,
                                 verbose=True),
      TerminateOnNaN(),
    ],
    max_epochs=CFG.epochs,
    val_check_interval=0.8,
    resume_from_checkpoint=best_path,
  )

  from pytorch_lightning.trainer.supporters import CombinedLoader
  names = ['a', 'p', 'n']
  train = CombinedLoader({k: v for k, v in zip(names, train)},
                         mode='max_size_cycle')
  valid = CombinedLoader({k: v for k, v in zip(names, valid)},
                         mode='max_size_cycle')
  trainer.fit(
    model,
    train_dataloaders=train,
    val_dataloaders=valid
  )


# ===========================================================================
# For evaluation
# ===========================================================================
def evaluate_covid_detector(model: torch.nn.Module):
  path, best_path = get_model_path(
    model, overwrite=False,
    monitor=CFG.load if len(CFG.load) > 0 else CFG.monitor)
  if best_path is None:
    raise RuntimeError(f'No model found at path: {path}')
  model = TrainModule.load_from_checkpoint(
    checkpoint_path=best_path,
    model=model,
    strict=False)
  # the pretrained model cannot be switch to CPU easily
  # model.cpu()
  model.eval()

  test_key = ['extra_train', 'final_pri_test', 'final_pub_test']
  pseudo_labels = dict()
  counts = 0
  with torch.no_grad():
    for key in test_key:
      if key not in ZIP_FILES:
        continue
      test = init_dataset(Partition(name=key), random_cut=-1,
                          is_training=False,
                          outputs=('signal', 'id'))
      results = dict()
      for batch in tqdm(to_loader(test, is_training=False, num_workers=2),
                        desc=key):
        y_proba = model(batch, proba=True).cpu().numpy()
        for k, v in zip(batch.id, y_proba):
          results[k] = v
          # pseudo-label
          counts += 1
          pseudo_labels[k] = v
      ## save to csv
      if '_test' not in key:  # no need for training data
        continue
      uuid_order = list(META_DATA[key].values())[0].uuid
      text = 'uuid,assessment_result\n'
      for k in uuid_order:
        text += f'{k},{results[k]}\n'
      csv_path = os.path.join(path, 'results.csv')
      zip_path = os.path.join(path, f'{key}.zip')
      with open(csv_path, 'w') as f:
        f.write(text[:-1])
      with zipfile.ZipFile(zip_path, 'w') as f:
        f.write(csv_path, arcname=os.path.basename(csv_path))
      print('Save results to:', zip_path)
      os.remove(csv_path)
  ## save pseudo-labeler
  print('Check overlap:', len(pseudo_labels), counts)
  outpath = os.path.join(PSEUDOLABEL_PATH, os.path.basename(path))
  with open(outpath, 'wb') as f:
    pickle.dump(pseudo_labels, f)
  print('Save pseudo-labels to', outpath)


def evaluate_agegen_recognizer(model: torch.nn.Module):
  path, best_path = get_model_path(
    model, overwrite=False,
    monitor=CFG.load if len(CFG.load) > 0 else CFG.monitor)
  if best_path is None:
    raise RuntimeError(f'No model found at path: {path}')
  model = TrainModule.load_from_checkpoint(
    checkpoint_path=best_path,
    model=model,
    strict=False)

  # generate pseudo-label
  results = dict()
  count = 0
  with torch.no_grad():
    model.eval()
    for key in ['extra_train', 'final_pub_test', 'final_pri_test']:
      ds = init_dataset(Partition(name=key),
                        is_training=False,
                        outputs=('signal', 'gender', 'age', 'id'))
      for batch in tqdm(to_loader(ds, num_workers=3, batch_size=CFG.bs,
                                  is_training=False),
                        desc=key):
        age_pred, gen_pred = model(batch, proba=True)
        age_true, gen_true = batch.age.data, batch.gender.data
        for uuid, ap, at, gp, gt in zip(batch.id,
                                        age_pred, age_true,
                                        gen_pred, gen_true):
          count += 1
          ap = ap.cpu().numpy().tolist()
          at = at.numpy().tolist()
          gp = gp.cpu().numpy().tolist()
          gt = gt.numpy().tolist()
          results[uuid] = (at if at >= 0 else ap, gt if gt >= 0 else gp)
  print('Check overlap uuid:', len(results), count)
  outpath = os.path.join(PSEUDO_AGEGEN, os.path.basename(path))
  with open(outpath, 'wb') as f:
    pickle.dump(results, f)
  print('Saved age and gender labels to:', outpath)


# ===========================================================================
# Main
# ===========================================================================
def main():
  # CFG.pseudolabel = True
  # CFG.mixup = True
  # CFG.task = 'contrastive'
  # ds = init_dataset(Partition('final_train'),
  #                   outputs=('signal', 'result', 'gender', 'age'))
  # for i in tqdm(ds):
  #   r = i['result']
  #   g = i['gender']
  #   a = i['age']
  # exit()
  train_percent = 0.8
  train_ds = Partition(name='final_train', start=0.0, end=train_percent)
  if CFG.pseudolabel:
    train_ds = [train_ds, Partition(name='extra_train',
                                    start=0.0,
                                    end=1.0)]
  valid_ds = Partition(name='final_train', start=train_percent, end=1.0)

  ## create the dataset
  if CFG.task == 'covid':
    outputs = ('signal', 'result', 'age', 'gender')
    train = init_dataset(train_ds, is_training=True,
                         random_cut=CFG.random_cut, outputs=outputs)
    valid = init_dataset(valid_ds, is_training=False,
                         random_cut=-1, outputs=outputs)
  elif CFG.task == 'contrastive':
    outputs = ('signal', 'result', 'age', 'gender')
    kw = dict(partition=train_ds, random_cut=CFG.random_cut, outputs=outputs,
              is_training=True)
    train_anchor = init_dataset(only_result=1, **kw)
    train_pos = init_dataset(only_result=1, **kw)
    train_neg = init_dataset(only_result=0, **kw)

    kw = dict(partition=valid_ds, random_cut=CFG.random_cut, outputs=outputs,
              is_training=False)
    valid_anchor = init_dataset(only_result=1, **kw)
    valid_pos = init_dataset(only_result=1, **kw)
    valid_neg = init_dataset(only_result=0, **kw)
  elif CFG.task == 'gender':
    outputs = ('signal', 'gender', 'age')
    train = init_dataset(train_ds, random_cut=CFG.random_cut, outputs=outputs,
                         is_training=True)
    valid = init_dataset(valid_ds, random_cut=-1, outputs=outputs,
                         is_training=False)
  else:
    raise NotImplementedError(f'No support for task={CFG.task}')

  ## create the model
  fn_model = globals().get(CFG.model, None)
  if fn_model is None:
    defined_models = []
    for k, v in globals().items():
      if inspect.isfunction(v):
        spec = inspect.getfullargspec(v)
        if 'return' in spec.annotations:
          return_type = spec.annotations['return']
          if isinstance(return_type, type) and \
              issubclass(return_type, torch.nn.Module):
            defined_models.append(k)
    print('Found defined models are:')
    for m in defined_models:
      print(f' - {m}')
    raise ValueError(f'Cannot find model with name="{CFG.model}", '
                     f'model is the name of a function '
                     f'defined in models.py')
  model = fn_model(CFG)
  # assign model.name (important for saving path)
  model.name = CFG.model

  ## save the config
  # only overwrite here
  path, _ = get_model_path(
    model, overwrite=True,
    monitor=CFG.load if len(CFG.load) > 0 else CFG.monitor)
  cfg_path = os.path.join(path, 'cfg.yaml')
  with open(cfg_path, 'w') as f:
    print('Save config to path:', cfg_path)
    for k, v in CFG.__dict__.items():
      f.write(f'{k}:{v}\n')

  ## running the task
  if CFG.eval:
    if CFG.task == 'covid':
      evaluate_covid_detector(model)
    elif CFG.task == 'gender':
      evaluate_agegen_recognizer(model)
    else:
      raise NotImplementedError(
        f'No support for evaluation mode task="{CFG.task}"')
  else:
    if CFG.task == 'covid':
      train_covid_detector(model, train, valid=valid, target='result')
    elif CFG.task == 'contrastive':
      train_contrastive(model,
                        train=[train_anchor, train_pos, train_neg],
                        valid=[valid_anchor, valid_pos, valid_neg])
    elif CFG.task == 'gender':
      train_covid_detector(model, train, valid=valid, target=['age', 'gender'])
    else:
      raise NotImplementedError(
        f'No support for training mode task="{CFG.task}"')


# ===========================================================================
# Main
# ===========================================================================
def _read_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('--overwrite', action='store_true')
  parser.add_argument('--eval', action='store_true')
  for k, v in Config.__annotations__.items():
    if k in ('eval', 'overwrite'):
      continue
    if v is bool:
      parser.add_argument(f'--{k}', action='store_true')
    else:
      parser.add_argument(f'-{k}', type=v, default=Config.__dict__[k])
  parsed_args: argparse.Namespace = parser.parse_args()
  for k, v in parsed_args.__dict__.items():
    if hasattr(CFG, k):
      # all argument is case insensitive
      if isinstance(v, str):
        v = v.lower()
      setattr(CFG, k, v)
  print('Read arguments:')
  for k, v in CFG.__dict__.items():
    print(' - ', k, ':', v)
  # rescale the steps according to batch size
  CFG.steps_priming = int(CFG.steps_priming / (CFG.bs / 16))
  if CFG.eval:
    CFG.mixup = False
    CFG.pseudolabel = False
    CFG.overwrite = False


if __name__ == '__main__':
  _read_arguments()
  try:
    main()
  except Exception as e:
    p = write_errors(str(e), str(CFG))
    with open(p, 'a') as f:
      traceback.print_exc(file=f)
    raise e
