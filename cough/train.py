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
import dataclasses
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

import pytorch_lightning as pl
import torch.nn
from sklearn.metrics import auc as auc_score, roc_curve, confusion_matrix, \
  f1_score
from speechbrain.dataio.dataloader import SaveableDataLoader
from speechbrain.dataio.dataset import DynamicItemDataset
from speechbrain.dataio.sampler import ReproducibleWeightedRandomSampler, \
  ReproducibleRandomSampler
from torch.optim import lr_scheduler
from tqdm import tqdm

from config import META_DATA, get_json, SAVE_PATH, POS_WEIGHT, ZIP_FILES, \
  DATA_SEED, PSEUDOLABEL_PATH
from features import AudioRead, VAD, LabelEncoder, PseudoLabeler
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
  # vad, energies
  ds.add_dynamic_item(VAD(), takes=VAD.takes, provides=VAD.provides)
  # result, gender, age
  ds.add_dynamic_item(LabelEncoder(pseudo_labeling=CFG.pseudolabel,
                                   pseudo_rand=CFG.pseudorand,
                                   pseudo_soft=CFG.pseudosoft),
                      takes=LabelEncoder.takes,
                      provides=LabelEncoder.provides)
  ds.set_output_keys(outputs)
  return ds


def to_loader(ds: DynamicItemDataset, num_workers: int = 0,
              is_training=True, drop_last=False):
  sampler = None
  if is_training:
    if CFG.oversampling:
      # oversampling with replacement
      weights = {}
      counts = defaultdict(int)
      for v in ds.data.values():
        meta = v['meta']
        uuid = meta['uuid']
        weights[uuid] = meta['assessment_result']
        counts[meta['assessment_result']] += 1
      # normalize
      counts = {k: sum(i for i in counts.values()) / v
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
  return SaveableDataLoader(
    dataset=ds,
    sampler=sampler,
    collate_fn=PaddedBatch,
    num_workers=num_workers,
    drop_last=drop_last,
    batch_size=CFG.bs if is_training else 8,
    pin_memory=True if torch.cuda.is_available() else False,
  )


def get_model_path(model, overwrite=False) -> Tuple[str, str]:
  overwrite = CFG.overwrite & overwrite
  monitor = CFG.monitor
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
  if len(checkpoints) > 0:
    for k in [monitor, 'val_loss']:
      if k in checkpoints[0]:
        key = k
        break
    if 'loss' in key:
      reverse = False  # smaller better
    else:
      reverse = True  # higher better
    best_path = sorted(
      checkpoints,
      key=lambda p: float(
        next(re.finditer(f'{key}=\d+\.\d+', p)).group().split('=')[-1]),
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
               target: str = 'result'):
    super().__init__()
    self.model = model
    self.fn_bce = torch.nn.BCEWithLogitsLoss(
      pos_weight=torch.tensor(CFG.pos_weight_rescale * POS_WEIGHT))
    self.fn_ce = torch.nn.CrossEntropyLoss()
    self.target = target
    self.label_noise = float(CFG.label_noise)
    self.lr = float(CFG.lr)

  def forward(self, x: PaddedBatch, proba: bool = False):
    # in lightning, forward defines the prediction/inference actions
    y = self.model(x)
    if proba:
      y = torch.sigmoid(y)
    return y

  def training_step(self, batch, batch_idx):
    y_pred = self.model(batch)
    if isinstance(y_pred, ModelOutput):
      extra_losses = y_pred.losses
      y_pred = y_pred.outputs
    else:
      extra_losses = 0.
    y_pred = y_pred.float()
    y_true = getattr(batch, self.target).data
    if torch.cuda.is_available():
      y_true.cuda()
    if len(y_pred.shape) == 1:
      n_target = 2
      y_true = y_true.float()
      if self.label_noise is not None and self.label_noise > 0:
        z = torch.rand(y_true.shape,
                       device=y_true.get_device()) * self.label_noise
        y_true = y_true * (1 - z) + (1 - y_true) * z
      loss = self.fn_bce(y_pred, y_true)
    else:
      n_target = y_pred.shape[-1]
      loss = self.fn_ce(y_pred, y_true.long())
    # Logging to TensorBoard by default
    loss += extra_losses
    self.log("train_loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    y = self.model(batch)
    if isinstance(y, ModelOutput):
      y = y.outputs
    if len(y.shape) == 1:
      y_pred = torch.sigmoid(y).ge(0.5)
    else:
      y_pred = torch.argmax(y, -1)
    y_true = getattr(batch, self.target).data.float().cuda()
    acc = y_pred.eq(y_true).float().sum() / y_pred.shape[0]
    return dict(acc=acc, true=y_true, pred=y_pred)

  def validation_epoch_end(self, outputs):
    acc = np.mean([o['acc'].cpu().numpy() for o in outputs])
    # calculate AUC
    true = torch.cat([o['true'] for o in outputs], 0).cpu().numpy()
    pred = torch.cat([o['pred'] for o in outputs], 0).cpu().numpy()
    # f1 score
    f1 = f1_score(true, pred)
    if np.isnan(f1) or np.isinf(f1):
      f1 = 0.
    # row: true_labels
    print(f'\n\n{confusion_matrix(true, pred)}\n')
    fpr, tpr, thresholds = roc_curve(true, pred, pos_label=1)
    auc = auc_score(fpr, tpr)
    if np.isnan(auc) or np.isinf(auc):
      auc = 0
    self.log('val_acc', torch.tensor(acc), prog_bar=True)
    self.log('val_auc', torch.tensor(auc), prog_bar=True)
    self.log('val_f1', torch.tensor(f1), prog_bar=True)
    # print the learning rate
    for pg in self.optimizers().param_groups:
      self.log('lr', pg.get('lr'), prog_bar=True)
    return dict(acc=acc, auc=auc)

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
                         target: str = 'result',
                         monitor: str = 'val_f1'):
  path, best_path = get_model_path(model, overwrite=False)

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
      pl.callbacks.EarlyStopping(monitor,
                                 mode='max',
                                 patience=CFG.patience,
                                 verbose=True),
      TerminateOnNaN(),
    ],
    max_epochs=CFG.epochs,
    val_check_interval=int(200 / (CFG.bs / 16)),
    resume_from_checkpoint=best_path,
  )

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
  path, best_path = get_model_path(model, overwrite=False)
  # no need oversampling
  CFG.oversampling = False

  train = [to_loader(i, num_workers=max(1, CFG.ncpu // 2),
                     is_training=True, drop_last=True)
           for i in train]
  valid = [to_loader(i, num_workers=0, is_training=False, drop_last=True)
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
                                   save_top_k=20,
                                   verbose=True),
      pl.callbacks.EarlyStopping('val_loss',
                                 mode='min',
                                 patience=CFG.patience,
                                 verbose=True),
      TerminateOnNaN(),
    ],
    max_epochs=CFG.epochs,
    val_check_interval=0.5,
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
  path, best_path = get_model_path(model, overwrite=False)
  if best_path is None:
    raise RuntimeError(f'No model found at path: {path}')
  model = TrainModule.load_from_checkpoint(
    checkpoint_path=best_path,
    model=model)
  # the pretrained model cannot be switch to CPU easily
  # model.cpu()
  model.eval()

  test_key = ['final_pub_test', 'final_pri_test']
  with torch.no_grad():
    for key in test_key:
      if key not in ZIP_FILES:
        continue
      test = init_dataset(Partition(name=key),
                          random_cut=-1,
                          outputs=('signal', 'id'))
      results = dict()
      for batch in tqdm(to_loader(test,
                                  is_training=False,
                                  num_workers=0),
                        desc=key):
        y_proba = model(batch, proba=True).cpu().numpy()
        for k, v in zip(batch.id, y_proba):
          results[k] = v
      uuid_order = list(META_DATA['final_pub_test'].values())[0].uuid
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


def pseudo_labeling(model: torch.nn.Module):
  path, best_path = get_model_path(model, overwrite=False)
  if best_path is None:
    raise RuntimeError(f'No model found at path: {path}')
  model = TrainModule.load_from_checkpoint(
    checkpoint_path=best_path,
    model=model)
  # the pretrained model cannot be switch to CPU easily
  # model.cpu()
  model.eval()

  test_key = ['final_train', 'extra_train',
              'final_pub_test', 'final_pri_test']
  labels = dict()
  n = 0
  with torch.no_grad():
    for key in test_key:
      if key not in ZIP_FILES:
        continue
      test = init_dataset(Partition(name=key),
                          random_cut=-1,
                          outputs=('signal', 'id'))
      for batch in tqdm(to_loader(test, is_training=False,
                                  num_workers=0),
                        desc=key):
        y_proba = model(batch, proba=True).cpu().numpy()
        for k, v in zip(batch.id, y_proba):
          n += 1
          labels[k] = v
  outpath = os.path.join(PSEUDOLABEL_PATH, os.path.basename(path))
  with open(outpath, 'wb') as f:
    pickle.dump(labels, f)
  print('Save pseudo-labels to', outpath)


# ===========================================================================
# Main
# ===========================================================================
def main():
  # print(pretrained_sepformer().modules)
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
    train = init_dataset(train_ds, random_cut=CFG.random_cut, outputs=outputs)
    valid = init_dataset(valid_ds, random_cut=-1, outputs=outputs)
  elif CFG.task == 'contrastive':
    outputs = ('signal', 'result', 'age', 'gender')
    kw = dict(partition=train_ds, random_cut=CFG.random_cut, outputs=outputs)
    train_anchor = init_dataset(only_result=1, **kw)
    train_pos = init_dataset(only_result=1, **kw)
    train_neg = init_dataset(only_result=0, **kw)

    kw = dict(partition=valid_ds, random_cut=CFG.random_cut, outputs=outputs)
    valid_anchor = init_dataset(only_result=1, **kw)
    valid_pos = init_dataset(only_result=1, **kw)
    valid_neg = init_dataset(only_result=0, **kw)
  elif CFG.task == 'pseudolabel':
    pass
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
  path, _ = get_model_path(model, overwrite=True)
  cfg_path = os.path.join(path, 'cfg.yaml')
  with open(cfg_path, 'w') as f:
    print('Save config to path:', cfg_path)
    for k, v in CFG.__dict__.items():
      f.write(f'{k}:{v}\n')

  ## running the task
  if CFG.eval:
    if CFG.task == 'covid':
      evaluate_covid_detector(model)
    else:
      raise NotImplementedError(
        f'No support for evaluation mode task="{CFG.task}"')
  else:
    if CFG.task == 'covid':
      train_covid_detector(model, train, valid=valid, target='result')
    elif CFG.task == 'pseudolabel':
      pseudo_labeling(model)
    elif CFG.task == 'contrastive':
      train_contrastive(model,
                        train=[train_anchor, train_pos, train_neg],
                        valid=[valid_anchor, valid_pos, valid_neg])
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
  if CFG.eval or CFG.task == 'pseudolabel':
    CFG.overwrite = False


if __name__ == '__main__':
  _read_arguments()
  main()
