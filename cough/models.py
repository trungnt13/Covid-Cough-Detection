from typing import Union, Callable, List

import speechbrain
from speechbrain.nnet.containers import Sequential
from speechbrain.nnet.normalization import BatchNorm1d
from speechbrain.nnet.linear import Linear
from speechbrain.core import Brain
import torch
from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier, \
  SepformerSeparation, SpectralMaskEnhancement, EncoderDecoderASR
from speechbrain.dataio.dataloader import PaddedBatch
from config import dev


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
# Basic model
# ===========================================================================
PretrainedModel = Union[
  EncoderClassifier, EncoderDecoderASR, SepformerSeparation, SpectralMaskEnhancement]


class Classifier(Sequential):

  def __init__(
      self,
      input_shape,
      activation=torch.nn.LeakyReLU,
      lin_blocks=2,
      lin_neurons=512,
      out_neurons=1,
  ):
    super().__init__(input_shape=input_shape)

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


class SimpleClassifier(torch.nn.Module):

  def __init__(self, features: List[PretrainedModel]):
    super(SimpleClassifier, self).__init__()
    features = list(features)
    # infer the input shape
    x = torch.rand(5, 1000)
    input_shape = [f.encode_batch(x, wav_lens=torch.ones([5])).shape
                   for f in features]
    input_shape = list(input_shape[0][:-1]) + \
                  [sum(s[-1] for s in input_shape)]
    self.features = features
    self.classifier = Classifier(tuple(input_shape),
                                 lin_neurons=512,
                                 out_neurons=1)
    self.classifier.cuda()

  def forward(self, batch: PaddedBatch):
    signal = batch.signal.data
    lengths = batch.signal.lengths
    X = torch.cat(
      [f.encode_batch(signal, lengths) for f in self.features], -1)
    y = self.classifier(X).squeeze(-1).squeeze(-1)
    return y


def covid_brain():
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
