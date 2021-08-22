from typing import Union, Callable, List, Sequence, Optional

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

from config import dev, SAMPLE_RATE, Config
from logging import getLogger

logger = getLogger('models')


# ===========================================================================
# Prerained Model
# ===========================================================================


def pretrained_ecapa() -> EncoderClassifier:
  """[batch_size, 1, 192]"""
  print('Loading pretrained ECAPA-TDNN')
  return EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": dev()})


def pretrained_xvec() -> EncoderClassifier:
  """[batch_size, 1, 512]"""
  print('Loading pretrained X-vector')
  return EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    run_opts={"device": dev()})


def pretrained_langid() -> EncoderClassifier:
  """[batch_size, 1, 192]"""
  print('Loading pretrained Lang-ID')
  return EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-commonlanguage_ecapa",
    run_opts={"device": dev()})


def pretrained_wav2vec() -> EncoderDecoderASR:
  """[1, 2, 1024]"""
  print('Loading pretrained wav2vec EN')
  return EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-commonvoice-en",
    run_opts={"device": dev()})


def pretrained_wav2vec_chn() -> EncoderDecoderASR:
  print('Loading pretrained wav2vec CHN')
  return EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-transformer-aishell",
    run_opts={"device": dev()})


def pretrained_crdnn():
  print('Loading pretrained CRDNN')
  return EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-crdnn-transformerlm-librispeech",
    run_opts={"device": dev()})


def pretrained_transformer():
  print('Loading pretrained Transformer EN')
  return EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-transformerlm-librispeech",
    run_opts={"device": dev()})


def pretrained_transformer_chn() -> EncoderDecoderASR:
  print('Loading pretrained Transformer CHN')
  return EncoderDecoderASR.from_hparams(
    source="speechbrain/asr-transformer-aishell",
    run_opts={"device": dev()})


def pretrained_sepformer() -> SepformerSeparation:
  print('Loading pretrained Sepformer')
  return SepformerSeparation.from_hparams(
    source="speechbrain/sepformer-whamr",
    run_opts={"device": dev()})


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
               n_steps_priming: int = 1000):
    super(SimpleClassifier, self).__init__(features, name)
    self.n_steps_priming = int(n_steps_priming)
    self.augment = TimeDomainSpecAugment(
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
    self.classifier = Classifier(self._input_shape,
                                 dropout=dropout,
                                 lin_blocks=n_layers,
                                 lin_neurons=n_hidden,
                                 out_neurons=1 if n_target == 2 else n_target)
    if torch.cuda.is_available():
      self.classifier.cuda()

  def pretrained_parameters(self, named: bool = False):
    return sum([list(f.modules.named_parameters()
                     if named else f.modules.parameters())
                for f in self.features], [])

  def forward(self, batch: PaddedBatch):
    signal = batch.signal.data
    lengths = batch.signal.lengths
    if self.training:
      # data augmentation
      signal = self.augment(signal, lengths)
      self.training_steps += 1
      # enable pretrained parameters
      if self.training_stage == 0 and \
          self.training_steps > self.n_steps_priming:
        logger.info(f'[{self.name}] Enable all pretrained parameters')
        print(f'\n[{self.name}] Enable all pretrained parameters')
        self.set_pretrained_params(True)
        self.training_stage += 1
    try:
      # feature extraction
      X = torch.cat(
        [f.encode_batch(signal, lengths) for f in self.features], -1)
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
    n_steps_priming=int(cfg.steps_priming / (cfg.bs / 16)))
  return model
