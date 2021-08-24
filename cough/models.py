import dataclasses
import warnings
from typing import Union, Callable, List, Sequence, Optional

import numpy as np
import speechbrain
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.linear import Linear
from speechbrain.core import Brain
import torch
from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier, \
  SepformerSeparation, SpectralMaskEnhancement, EncoderDecoderASR
from speechbrain.dataio.dataloader import PaddedBatch
from torch import nn
from speechbrain.nnet.pooling import StatisticsPooling

from config import dev, SAMPLE_RATE, Config, SEED
from logging import getLogger

logger = getLogger('models')


@dataclasses.dataclass()
class ModelOutput:
  outputs: Optional[torch.Tensor] = None
  losses: Optional[torch.Tensor] = None


# ===========================================================================
# Prerained Model
# ===========================================================================
def _load_encoder_classifier(source) -> EncoderClassifier:
  print(f'Loading pretrained {source}')
  m = EncoderClassifier.from_hparams(
    source=source,
    run_opts={"device": dev()})
  m.modules.pop('classifier')
  return m


def pretrained_ecapa() -> EncoderClassifier:
  """[batch_size, 1, 192]"""
  m = _load_encoder_classifier("speechbrain/spkrec-ecapa-voxceleb")
  return m


def pretrained_xvec() -> EncoderClassifier:
  """[batch_size, 1, 512]"""
  m = _load_encoder_classifier("speechbrain/spkrec-xvect-voxceleb")
  return m


def pretrained_langid() -> EncoderClassifier:
  """[batch_size, 1, 192]"""
  m = _load_encoder_classifier("speechbrain/lang-id-commonlanguage_ecapa")
  return m


###################### Wav2Vec
def _load_encoder_asr(source) -> EncoderDecoderASR:
  print(f'Loading pretrained {source}')
  m = EncoderDecoderASR.from_hparams(
    source=source,
    run_opts={"device": dev()})
  m.modules.pop('decoder')
  return m


def pretrained_wav2vec() -> EncoderDecoderASR:
  """[1, time_downsampled, 1024]"""
  return _load_encoder_asr("speechbrain/asr-wav2vec2-commonvoice-en")


def pretrained_wav2vec_chn() -> EncoderDecoderASR:
  return _load_encoder_asr("speechbrain/asr-wav2vec2-transformer-aishell")


def pretrained_crdnn():
  return _load_encoder_asr("speechbrain/asr-crdnn-transformerlm-librispeech")


def pretrained_transformer():
  return _load_encoder_asr(
    "speechbrain/asr-transformer-transformerlm-librispeech")


def pretrained_transformer_chn() -> EncoderDecoderASR:
  return _load_encoder_asr("speechbrain/asr-transformer-aishell")


###################### Sepformer
def pretrained_sepformer() -> SepformerSeparation:
  print('Loading pretrained Sepformer')
  m = SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-whamr",
    run_opts={"device": dev()})
  return m


####################### Speech Enhancer
def pretrained_metricgan() -> SpectralMaskEnhancement:
  print('Loading pretrained MetricGAN')
  return SpectralMaskEnhancement.from_hparams(
    source="speechbrain/metricgan-plus-voicebank",
    run_opts={"device": dev()})


def pretrained_mimic() -> SpectralMaskEnhancement:
  print('Loading pretrained MIMIC')
  return SpectralMaskEnhancement.from_hparams(
    source="speechbrain/mtl-mimic-voicebank",
    run_opts={"device": dev()})


# ===========================================================================
# Define model classes
# ===========================================================================
PretrainedModel = Union[EncoderClassifier,
                        EncoderDecoderASR,
                        SepformerSeparation,
                        SpectralMaskEnhancement]


class Classifier(Sequential):

  def __init__(
      self,
      input_shape: Sequence[int],
      activation: nn.Module = torch.nn.LeakyReLU,
      dropout: Optional[float] = 0.3,
      lin_blocks: int = 2,
      lin_neurons: int = 512,
      out_neurons: int = 1,
  ):
    super().__init__(input_shape=input_shape)
    if dropout is not None and 0. < dropout < 1.:
      self.append(nn.Dropout(p=dropout), layer_name='drop')

    self.append(activation(), layer_name="act")
    self.append(BatchNorm1d, layer_name="norm")

    if lin_blocks > 0:
      self.append(Sequential, layer_name="DNN")

    for block_index in range(lin_blocks):
      block_name = f"block_{block_index}"
      self.DNN.append(Sequential, layer_name=block_name)
      self.DNN[block_name].append(
        Linear,
        n_neurons=lin_neurons,
        bias=True,
        layer_name="linear",
      )
      self.DNN[block_name].append(activation(), layer_name="act")
      self.DNN[block_name].append(
        BatchNorm1d, layer_name="norm"
      )

    # Final Softmax classifier
    self.append(
      Linear, n_neurons=out_neurons, layer_name="out"
    )


class CoughModel(torch.nn.Module):

  def __init__(self, features: List[PretrainedModel], name: str = None):
    super().__init__()
    features = list(features)
    # infer the input shape
    x = torch.rand(5, 1000)
    input_shape = [f.encode_batch(x, wav_lens=torch.ones([5])).shape
                   for f in features]
    input_shape = list(input_shape[0][:-1]) + \
                  [sum(s[-1] for s in input_shape)]
    if input_shape[1] != 1:
      warnings.warn(f'Pretrained model returned non-1 time dimension '
                    f'{input_shape}, need pooling!')
      input_shape[1] = 1
      input_shape[-1] *= 2
    self._input_shape = tuple(input_shape)
    self.features = features
    self._pretrained = torch.nn.ModuleList([f.modules for f in self.features])
    self.name = name
    # set require_grad=False for pretrained parameters
    self.set_pretrained_params(trainable=False)
    self.training_stage = 0
    self.training_steps = 0

  def set_pretrained_params(self, trainable) -> 'CoughModel':
    for p in self.pretrained_parameters():
      p: nn.Parameter
      p.requires_grad = trainable
    return self

  def pretrained_parameters(self, named: bool = False):
    return sum([list(f.modules.named_parameters()
                     if named else f.modules.parameters())
                for f in self.features], [])


class SimpleClassifier(CoughModel):

  def __init__(self,
               features: List[PretrainedModel],
               name: str = None,
               dropout: float = 0.3,
               n_hidden: int = 512,
               n_layers: int = 2,
               n_target: int = 2,
               n_steps_priming: int = 1000,
               n_inputs_classifier: int = 1):
    super(SimpleClassifier, self).__init__(features, name)
    self.dropout = dropout
    self.n_steps_priming = int(n_steps_priming)
    self.n_hidden = n_hidden
    self.n_layers = n_layers
    self.n_target = n_target

    self.augmenter = TimeDomainSpecAugment(
      perturb_prob=0.8,
      drop_freq_prob=0.8,
      drop_chunk_prob=0.8,
      speeds=[90, 95, 100, 105, 110],
      sample_rate=SAMPLE_RATE,
      drop_freq_count_low=2,
      drop_freq_count_high=5,
      drop_chunk_count_low=2,
      drop_chunk_count_high=8,
      drop_chunk_length_low=1000,
      drop_chunk_length_high=SAMPLE_RATE,
      drop_chunk_noise_factor=0.1)

    shape = self._input_shape[:-1] + \
            (self._input_shape[-1] * n_inputs_classifier,)
    self.classifier = Classifier(shape,
                                 dropout=dropout,
                                 lin_blocks=n_layers,
                                 lin_neurons=n_hidden,
                                 out_neurons=1 if n_target == 2 else n_target)
    if torch.cuda.is_available():
      self.classifier.cuda()

    self.stats_pooling = StatisticsPooling()

  def encode(self, signal, lengths):
    X = torch.cat(
      [f.encode_batch(signal, lengths) for f in self.features], -1)
    # statistical pooling
    if X.shape[1] != 1:
      X = self.stats_pooling(X)
    return X

  def augment(self, signal, lengths):
    if self.training:
      # data augmentation
      signal = self.augmenter(signal, lengths)
      self.training_steps += 1
      # enable pretrained parameters
      if self.training_stage == 0 and \
          self.training_steps > self.n_steps_priming:
        logger.info(f'[{self.name}] Enable all pretrained parameters')
        print(f'\n[{self.name}] Enable all pretrained parameters')
        self.set_pretrained_params(trainable=True)
        self.training_stage += 1
    return signal

  def forward(self, batch: PaddedBatch, return_feat: bool = False):
    signal = batch.signal.data
    lengths = batch.signal.lengths
    signal = self.augment(signal, lengths)
    try:
      # feature extraction
      X = self.encode(signal, lengths)
      # classifier
      y = self.classifier(X).squeeze(1)
      if y.shape[-1] == 1:
        y = y.squeeze(-1)
    except RuntimeError as e:
      print('signals:', signal.get_device())
      print('lengths:', lengths.get_device())
      for name, p in self.pretrained_parameters(named=True):
        print(name, p.shape, p.get_device())
      for name, p in self.named_parameters():
        print(name, p.shape, p.get_device())
      raise e
    if return_feat:
      return y, X
    return y


def _match_len(signal, min_len, rand):
  l = signal.shape[1]
  if l > min_len:
    i = rand.randint(0, l - min_len - 1)
    signal = signal[:, i: i + min_len]
  return signal


class ContrastiveLearner(SimpleClassifier):

  def __init__(self, *args, **kwargs):
    super(ContrastiveLearner, self).__init__(n_inputs_classifier=2,
                                             *args,
                                             **kwargs)
    self.rand = np.random.RandomState(SEED)
    self.fn_bce = nn.BCEWithLogitsLoss()

  def forward(self,
              anchor: PaddedBatch,
              positive: PaddedBatch,
              negative: PaddedBatch):
    a, a_l = anchor.signal.data, anchor.signal.lengths
    p, p_l = positive.signal.data, positive.signal.lengths
    n, n_l = negative.signal.data, negative.signal.lengths
    # cut all three to even length
    min_len = min(a.shape[1], p.shape[1], n.shape[1])
    a = _match_len(a, min_len=min_len, rand=self.rand)
    p = _match_len(p, min_len=min_len, rand=self.rand)
    n = _match_len(n, min_len=min_len, rand=self.rand)

    # augment and get embedding
    signal_clean = torch.cat([a, p, n], 0)
    lengths = torch.cat([a_l, p_l, n_l], 0)
    signal_aug = self.augment(signal_clean, lengths)

    emb_clean = self.encode(signal_clean, lengths)
    emb_aug = self.encode(signal_aug, lengths)
    emb_clean_anchor, emb_clean_pos, emb_clean_neg = emb_clean.chunk(3)
    emb_aug_anchor, emb_aug_pos, emb_aug_neg = emb_aug.chunk(3)

    # If I switch anchor + positive sample I have another positive
    # sample for free
    positive_clean = torch.cat([emb_clean_anchor, emb_clean_pos], dim=2)
    positive_clean2 = torch.cat([emb_clean_pos, emb_clean_anchor], dim=2)
    positive_noise = torch.cat([emb_aug_anchor, emb_aug_pos], dim=2)
    positive_noise2 = torch.cat([emb_aug_pos, emb_aug_anchor], dim=2)
    # Combining clean and noisy samples as well
    positive_mix = torch.cat([emb_clean_anchor, emb_aug_pos], dim=2)
    positive_mix2 = torch.cat([emb_clean_pos, emb_aug_anchor], dim=2)
    positive = torch.cat([positive_clean,
                          positive_noise,
                          positive_mix,
                          positive_clean2,
                          positive_noise2,
                          positive_mix2])

    # Note: If I switch anchor + negative sample I have another negative
    # sample for free
    negative_clean = torch.cat([emb_clean_anchor, emb_clean_neg], dim=2)
    negative_clean2 = torch.cat([emb_clean_neg, emb_clean_anchor], dim=2)
    negative_noise = torch.cat([emb_aug_anchor, emb_aug_neg], dim=2)
    negative_noise2 = torch.cat([emb_aug_neg, emb_aug_anchor], dim=2)
    # Combining clean and noisy samples as well
    negative_mix = torch.cat([emb_clean_anchor, emb_aug_neg], dim=2)
    negative_mix2 = torch.cat([emb_clean_neg, emb_aug_anchor], dim=2)
    negative = torch.cat([negative_clean,
                          negative_noise,
                          negative_mix,
                          negative_clean2,
                          negative_noise2,
                          negative_mix2])
    # contrastive loss using BCE
    samples = torch.cat([positive, negative])
    targets = torch.cat([torch.ones(positive.shape[0], device=positive.device),
                         torch.zeros(positive.shape[0], device=positive.device),
                         ])
    targets = targets.unsqueeze(1).unsqueeze(1)
    outputs = self.classifier(samples)
    loss = self.fn_bce(outputs, targets)
    return loss


class DomainBackprop(SimpleClassifier):
  """ Gamin et al. Unsupervised Domain Adaptation by Backpropagation. 2019 """

  def __init__(self,
               age_coef=0.05,
               gen_coef=0.05,
               decay_rate=0.98,
               step_size=100,
               min_coef=1e-10,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.age_classifier = Classifier(
      input_shape=self._input_shape,
      dropout=self.dropout,
      lin_blocks=self.n_layers,
      lin_neurons=self.n_hidden,
      out_neurons=10)
    self.gender_classifier = Classifier(
      input_shape=self._input_shape,
      dropout=self.dropout,
      lin_blocks=self.n_layers,
      lin_neurons=self.n_hidden,
      out_neurons=3)
    self.fn_loss = nn.CrossEntropyLoss()
    self.age_coef = age_coef
    self.gen_coef = gen_coef
    self.decay_rate = float(decay_rate)
    self.step_size = step_size
    self.n_steps = 0
    self.min_coef = min_coef

  def forward(self, batch: PaddedBatch):
    y, X = super(DomainBackprop, self).forward(batch, return_feat=True)
    if self.training:
      age = self.age_classifier(X).squeeze(1)
      gen = self.gender_classifier(X).squeeze(1)
      age_true = batch.age.data
      gen_true = batch.gender.data
      # maximizing the loss here
      decay_rate = self.decay_rate ** int(self.n_steps / self.step_size)
      self.n_steps += 1
      losses: torch.Tensor = -(
          max(decay_rate * self.age_coef, self.min_coef) *
          self.fn_loss(age, age_true) +
          max(decay_rate * self.gen_coef, self.min_coef) *
          self.fn_loss(gen, gen_true))
      return ModelOutput(outputs=y, losses=losses)
    return y


# ===========================================================================
# Defined model
# ===========================================================================
def simple_xvec(cfg: Config) -> CoughModel:
  features = [pretrained_xvec()]
  model = SimpleClassifier(
    features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def contrastive_xvec(cfg: Config) -> ContrastiveLearner:
  features = [pretrained_xvec()]
  model = ContrastiveLearner(
    features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def simple_ecapa(cfg: Config) -> CoughModel:
  features = [pretrained_ecapa()]
  model = SimpleClassifier(
    features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def wav2vec_en(cfg: Config) -> CoughModel:
  features = [pretrained_wav2vec()]
  model = SimpleClassifier(
    features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def wav2vec_chn(cfg: Config) -> CoughModel:
  features = [pretrained_wav2vec_chn()]
  model = SimpleClassifier(
    features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def transformer_en(cfg: Config) -> CoughModel:
  features = [pretrained_transformer()]
  model = SimpleClassifier(
    features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def transformer_chn(cfg: Config) -> CoughModel:
  features = [pretrained_transformer_chn()]
  model = SimpleClassifier(
    features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def domain_xvec(cfg: Config) -> CoughModel:
  features = [pretrained_xvec()]
  age = 0.05
  gen = 0.05
  args = [i for i in cfg.model_args.split(',') if len(i) > 0]
  if len(args) == 1:
    age = float(args[0])
    gen = float(args[0])
  elif len(args) > 1:
    age = float(args[0])
    gen = float(args[1])
  model = DomainBackprop(
    age_coef=age,
    gen_coef=gen,
    step_size=int(100 / (cfg.bs / 16)),
    features=features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def sepformer(cfg: Config) -> CoughModel:
  features = [pretrained_sepformer()]
  print(features[0].modules)
  exit()
