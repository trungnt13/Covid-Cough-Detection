import dataclasses
import os
import warnings
from typing import Union, Callable, List, Sequence, Optional

import numpy as np
import speechbrain
import torchaudio.backend.soundfile_backend
from speechbrain.lobes.augment import TimeDomainSpecAugment
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.linear import Linear
from speechbrain.core import Brain
import torch
from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier, \
  SepformerSeparation, SpectralMaskEnhancement, EncoderDecoderASR
from speechbrain.dataio.dataloader import PaddedBatch
from speechbrain.processing.speech_augmentation import AddNoise
from torch import nn
from speechbrain.nnet.pooling import StatisticsPooling

from config import dev, SAMPLE_RATE, Config, SEED, WAV_FILES, CACHE_PATH, \
  GEN_WEIGHT, AGE_WEIGHT
from logging import getLogger

from transformers.models.wav2vec2.modeling_wav2vec2 import \
  Wav2Vec2FeatureExtractor, Wav2Vec2Config, Wav2Vec2Model, \
  Wav2Vec2FeatureProjection, Wav2Vec2Encoder
from torch.autograd import Function

logger = getLogger('models')


@dataclasses.dataclass()
class ModelOutput:
  outputs: Optional[torch.Tensor] = None
  losses: Optional[torch.Tensor] = None


class GradientReverseF(Function):
  """ Credit: https://github.com/fungtion/DANN """

  @staticmethod
  def forward(ctx, x, alpha):
    ctx.alpha = alpha
    return x.view_as(x)

  @staticmethod
  def backward(ctx, grad_output):
    output = grad_output.neg() * ctx.alpha
    return output, None


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


class Pitch2Vec(torch.nn.Module):

  def __init__(self):
    super(Pitch2Vec, self).__init__()
    config = Wav2Vec2Config(
      conv_dim=(32, 32, 32, 32),
      conv_stride=(5, 4, 3, 2),
      conv_kernel=(10, 8, 6, 4),
      hidden_size=512,
      feat_proj_dropout=0.1,
      num_attention_heads=8,
      num_hidden_layers=3)
    self.pitch_model = torch.nn.ModuleDict(
      dict(
        feature=Wav2Vec2FeatureExtractor(config),
        project=Wav2Vec2FeatureProjection(config),
        encoder=Wav2Vec2Encoder(config),
        final=nn.Sequential(nn.Linear(1024, 512),
                            nn.Linear(512, 512))
      )
    )
    self.stats_pooling = StatisticsPooling()

  def forward(self, pitch):
    pi = self.pitch_model.feature(pitch)
    pi = pi.transpose(1, 2)
    pi_h, pi = self.pitch_model.project(pi)
    pi = self.pitch_model.encoder(pi_h,
                                  attention_mask=None,
                                  output_attentions=None,
                                  output_hidden_states=None,
                                  return_dict=False)[0]
    pi = self.stats_pooling(pi)
    pi = self.pitch_model.final(pi)
    return pi


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


class MixNoise(AddNoise):

  def __init__(self,
               snr_low=0.1,
               snr_high=12,
               mix_prob=0.95):
    # # ID,duration,wav,wav_format
    # csv_path = os.path.join(CACHE_PATH, 'noise.csv')
    # with open(csv_path, 'w') as f:
    #   f.write(f'ID,duration,wav,wav_format\n')
    #   for path in WAV_FILES['extra_train']:
    #     info = torchaudio.info(path)
    #     duration = info.num_frames / info.sample_rate
    #     uuid = os.path.basename(path).split('.')[0]
    #     f.write(f'{uuid},{duration:.2f},{path},wav\n')
    super(MixNoise, self).__init__(snr_low=snr_low,
                                   snr_high=snr_high,
                                   mix_prob=mix_prob,
                                   pad_noise=True,
                                   normalize=True)


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
               # these are for subclass configurations
               mix_noise: bool = True,
               snr_noise: float = 12.,
               perturb_prob=0.95,
               drop_freq_prob=0.95,
               drop_chunk_prob=0.95,
               speeds=(90, 95, 100, 105, 110),
               drop_freq_count_low=2,
               drop_freq_count_high=5,
               drop_chunk_count_low=2,
               drop_chunk_count_high=8,
               drop_chunk_length_low=1000,
               drop_chunk_length_high=2000,
               drop_chunk_noise_factor=0.,
               classifier_shape=None):
    super(SimpleClassifier, self).__init__(features, name)
    self.dropout = dropout
    self.n_steps_priming = int(n_steps_priming)
    self.n_hidden = n_hidden
    self.n_layers = n_layers
    self.n_target = n_target

    self.audio_mixer = MixNoise(snr_high=snr_noise)
    self.mix_audio = mix_noise

    self.augmenter = TimeDomainSpecAugment(
      perturb_prob=perturb_prob,
      drop_freq_prob=drop_freq_prob,
      drop_chunk_prob=drop_chunk_prob,
      speeds=speeds,
      sample_rate=SAMPLE_RATE,
      drop_freq_count_low=drop_freq_count_low,
      drop_freq_count_high=drop_freq_count_high,
      drop_chunk_count_low=drop_chunk_count_low,
      drop_chunk_count_high=drop_chunk_count_high,
      drop_chunk_length_low=drop_chunk_length_low,
      drop_chunk_length_high=drop_chunk_length_high,
      drop_chunk_noise_factor=drop_chunk_noise_factor)

    self.classifier = Classifier(
      self._input_shape if classifier_shape is None else classifier_shape,
      dropout=dropout,
      lin_blocks=n_layers,
      lin_neurons=n_hidden,
      out_neurons=1 if n_target == 2 else n_target)
    if torch.cuda.is_available():
      self.classifier.cuda()

    self.stats_pooling = StatisticsPooling()

  def encode(self, wavs, lengths):
    X = torch.cat(
      [f.encode_batch(wavs, lengths) for f in self.features], -1)
    # statistical pooling
    if X.shape[1] != 1:
      X = self.stats_pooling(X)
    return X

  def augment(self, wavs, lengths):
    if self.training:
      # data augmentation
      wavs = self.augmenter(wavs, lengths)
      if self.mix_audio:
        wavs = self.audio_mixer(wavs, lengths)
      self.training_steps += 1
      # enable pretrained parameters
      if self.training_stage == 0 and \
          self.training_steps > self.n_steps_priming:
        logger.info(f'[{self.name}] Enable all pretrained parameters')
        print(f'\n[{self.name}] Enable all pretrained parameters')
        self.set_pretrained_params(trainable=True)
        self.training_stage += 1
    return wavs

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


class SimpleGender(SimpleClassifier):

  def __init__(self, *args, **kwargs):
    features = [pretrained_xvec()]
    if 'dropout' in kwargs:
      dropout = kwargs.pop('dropout')
    else:
      dropout = 0.3
    emb_dim = 512
    super(SimpleGender, self).__init__(features=features,
                                       n_layers=1,
                                       n_hidden=1024,
                                       n_target=1,
                                       dropout=0.,
                                       classifier_shape=(5, 1, emb_dim),
                                       *args, **kwargs)
    self.age_classifier = Classifier(
      (5, 1, emb_dim),
      dropout=0,
      lin_blocks=1,
      lin_neurons=1024,
      out_neurons=1)
    if torch.cuda.is_available():
      self.age_classifier.cuda()

    self.pitch2vec = Pitch2Vec()

    if dropout > 0.:
      self.dropout = nn.Dropout(dropout)
    else:
      self.dropout = None

  def forward(self, batch: PaddedBatch, return_feat: bool = False):
    # wavs model
    # wavs = batch.signal.data
    # wavs_lengths = batch.signal.lengths
    # wavs = self.augment(wavs, wavs_lengths)
    # emb = self.encode(wavs, wavs_lengths)

    # pitch model
    pi = self.pitch2vec(batch.pitch.data)

    # final model
    # X = torch.cat([emb, pi], -1)
    X = pi
    if self.dropout is not None:
      X = self.dropout(X)
    gen = self.classifier(X).squeeze(1).squeeze(1)
    age = self.age_classifier(X).squeeze(1).squeeze(1)
    return age, gen


# ===========================================================================
# Contrastive learner
# ===========================================================================

def _match(wavs, lengths, min_batch, min_len, rand):
  l = wavs.shape[1]
  if l > min_len:
    i = rand.randint(0, l - min_len - 1)
    wavs = wavs[:min_batch, i: i + min_len]
    lengths = lengths[:min_batch]
  return wavs, lengths


class ContrastiveLearner(SimpleClassifier):

  def __init__(self, *args, **kwargs):
    super(ContrastiveLearner, self).__init__(
      mix_noise=True,
      snr_noise=20.,
      perturb_prob=0.98,
      drop_freq_prob=0.98,
      drop_chunk_prob=0.98,
      speeds=(85, 90, 95, 100, 105, 110, 115),
      drop_freq_count_low=2,
      drop_freq_count_high=8,
      drop_chunk_count_low=2,
      drop_chunk_count_high=6,
      drop_chunk_length_low=500,
      drop_chunk_length_high=4000,
      drop_chunk_noise_factor=0.,
      *args,
      **kwargs)
    self.rand = np.random.RandomState(SEED)
    self.fn_bce_reduce = nn.BCEWithLogitsLoss()
    self.fn_bce_none = nn.BCEWithLogitsLoss(reduction="none")

    shape = self._input_shape[:-1] + \
            (self._input_shape[-1] * 2,)
    self.discriminator = Classifier(
      input_shape=shape,
      dropout=self.dropout,
      lin_blocks=1,
      lin_neurons=2048,
      out_neurons=1
    )

  def forward(self,
              inputs,
              reduce: bool = True):
    if isinstance(inputs, PaddedBatch):
      return super(ContrastiveLearner, self).forward(inputs)

    anchor: PaddedBatch = inputs[0]
    positive: PaddedBatch = inputs[1]
    negative: PaddedBatch = inputs[2]

    a, a_l = anchor.signal.data, anchor.signal.lengths
    p, p_l = positive.signal.data, positive.signal.lengths
    n, n_l = negative.signal.data, negative.signal.lengths

    # cut all three to even length
    min_len = min(a.shape[1], p.shape[1], n.shape[1])
    min_batch = min(a.shape[0], p.shape[0], n.shape[0])
    a, a_l = _match(a, a_l, min_batch, min_len, rand=self.rand)
    p, p_l = _match(p, p_l, min_batch, min_len, rand=self.rand)
    n, n_l = _match(n, n_l, min_batch, min_len, rand=self.rand)

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
    outputs = self.discriminator(samples)
    if reduce:
      loss = self.fn_bce_reduce(outputs, targets)
    else:
      loss = self.fn_bce_none(outputs, targets)
    return loss


class DomainBackprop(SimpleClassifier):
  """ Gamin et al. Unsupervised Domain Adaptation by Backpropagation. 2019 """

  def __init__(self,
               coef=0.1,
               decay_rate=0.98,
               step_size=100,
               min_coef=1e-8,
               *args,
               **kwargs):
    super().__init__(*args, **kwargs)
    self.gender_classifier = Classifier(
      input_shape=self._input_shape,
      dropout=self.dropout,
      lin_blocks=self.n_layers,
      lin_neurons=self.n_hidden,
      out_neurons=1)
    self.age_classifier = Classifier(
      input_shape=self._input_shape,
      dropout=self.dropout,
      lin_blocks=self.n_layers,
      lin_neurons=self.n_hidden,
      out_neurons=1)
    self.gen_loss = nn.BCEWithLogitsLoss(pos_weight=GEN_WEIGHT)
    self.age_loss = nn.BCEWithLogitsLoss(pos_weight=AGE_WEIGHT)
    self.coef = coef
    self.decay_rate = float(decay_rate)
    self.step_size = step_size
    self.n_steps = 0
    self.min_coef = min_coef

  def forward(self, batch: PaddedBatch):
    y, emb = super(DomainBackprop, self).forward(batch, return_feat=True)
    if self.training:
      decay_rate = self.decay_rate ** int(self.n_steps / self.step_size)
      coef = max(decay_rate * self.coef, self.min_coef)
      revs_emb = GradientReverseF.apply(emb, coef)

      gen = self.gender_classifier(revs_emb).squeeze(1).squeeze(1)
      gen_true = batch.gender.data.float()

      age = self.age_classifier(revs_emb).squeeze(1).squeeze(1)
      age_true = batch.age.data.float()

      self.n_steps += 1

      losses = self.gen_loss(gen, gen_true) + self.age_loss(age, age_true)
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


def simple_gender(cfg: Config) -> CoughModel:
  model = SimpleGender(
    dropout=cfg.dropout,
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


def contrastive_ecapa(cfg: Config) -> ContrastiveLearner:
  features = [pretrained_ecapa()]
  model = ContrastiveLearner(
    features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def simple_langid(cfg: Config) -> CoughModel:
  features = [pretrained_langid()]
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
  coef = 0.1
  if len(cfg.model_args) > 0:
    coef = float(cfg.model_args)
  model = DomainBackprop(
    coef=coef,
    step_size=int(100 / (cfg.bs / 16)),
    features=features,
    dropout=cfg.dropout,
    n_target=2,
    n_steps_priming=cfg.steps_priming)
  return model


def domain_ecapa(cfg: Config) -> CoughModel:
  features = [pretrained_ecapa()]
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
