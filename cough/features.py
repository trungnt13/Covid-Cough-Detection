import glob
import os
import pickle
from collections import defaultdict
from random import random
from typing import Optional, Tuple, Any, Dict

import librosa
import numpy as np
import soundfile
import torch
import torchaudio
from torchaudio.functional import detect_pitch_frequency, compute_kaldi_pitch
from speechbrain.lobes.augment import TimeDomainSpecAugment, SpecAugment
from torch.nn import functional as F
from speechbrain.dataio.encoder import CategoricalEncoder
from speechbrain.pretrained import EncoderDecoderASR, EncoderClassifier, \
  SpectralMaskEnhancement
from scipy import stats
from config import SAMPLE_RATE, PSEUDOLABEL_PATH, SEED, write_errors, \
  PSEUDO_AGEGEN


def preemphasis(input: torch.tensor, coef: float = 0.97) -> torch.tensor:
  assert len(input.size()) == 1
  b = torch.FloatTensor([1.0, -coef])
  a = torch.FloatTensor([1.0, 0.0])
  y = torchaudio.functional.lfilter(input, a, b, clamp=True)
  return y


def vad_threshold(frames: torch.Tensor, threshold=35.) -> torch.Tensor:
  """
  threshold : scalar in range [30,50]
  """
  energies = 20 * torch.log10(torch.std(frames, 1) + 1e-10)
  max_energy = torch.max(energies)
  return torch.logical_and(energies > max_energy - threshold, energies > -75)


def map_fraction(s, e, duration, sr, offset, frames):
  length = ((e - s) * duration * sr) / frames
  s = (duration * s * sr - offset) / frames
  e = min(1.0, s + length)
  return dict(start=s, end=e)


_PSEUDO_LABELER = dict()


class PseudoLabeler:

  @staticmethod
  def get_labeler(pseudo_soft=False, pseudo_rand=False, pseudo_threshold=0.5):
    key = (pseudo_soft, pseudo_rand, pseudo_threshold)
    if key in _PSEUDO_LABELER:
      return _PSEUDO_LABELER[key]
    l = PseudoLabeler(pseudo_soft, pseudo_rand, pseudo_threshold)
    _PSEUDO_LABELER[key] = l
    return l

  def __init__(self,
               pseudo_soft=False,
               pseudo_rand=False,
               pseudo_threshold=0.5):
    key = (pseudo_soft, pseudo_rand, pseudo_threshold)
    if key in _PSEUDO_LABELER:
      raise ValueError('Call static method PseudoLabeler.get_labeler '
                       'for singleton.')
    self.rand = np.random.RandomState(SEED)
    self.pseudo_soft = pseudo_soft
    self.pseudo_rand = bool(pseudo_rand)
    self.pseudo_labels = []
    self.pseudo_threshold = pseudo_threshold
    for f in glob.glob(f'{PSEUDOLABEL_PATH}/*'):
      with open(f, 'rb') as f:
        print('Loaded Pseudo Labeler:', f)
        self.pseudo_labels.append(pickle.load(f))
    assert len(self.pseudo_labels) > 0, \
      f'No pseudo labels found at {PSEUDOLABEL_PATH}'

  def label(self, uuid) -> int:
    ####
    if self.pseudo_rand:
      label = self.rand.choice(self.pseudo_labels, 1)
      result = label[uuid]
      if not self.pseudo_soft:
        result = int(result > self.pseudo_threshold)
    ####
    else:
      if self.pseudo_soft:
        result = np.mean([i[uuid] for i in self.pseudo_labels])
      else:
        result = int(
          stats.mode([i[uuid] > self.pseudo_threshold
                      for i in self.pseudo_labels]).mode)
    return result


class PseudoAgeGen:

  @staticmethod
  def get_labeler():
    key = 'age-gen'
    if key not in _PSEUDO_LABELER:
      obj = PseudoAgeGen()
      _PSEUDO_LABELER[key] = obj
    return _PSEUDO_LABELER[key]

  def __init__(self):
    if 'age-gen' in _PSEUDO_LABELER:
      raise RuntimeError('call get_labeler for singleton')
    _PSEUDO_LABELER['age-gen'] = self

    self.pseudo_labels = []
    for f in glob.glob(f'{PSEUDO_AGEGEN}/*'):
      with open(f, 'rb') as f:
        print('Loaded Pseudo AgeGen:', f)
        self.pseudo_labels.append(pickle.load(f))
    assert len(self.pseudo_labels) > 0, \
      f'No pseudo labels found at {PSEUDO_AGEGEN}'

  def label(self, uuid, age, gen) -> Tuple[float, float]:
    if any(uuid not in i for i in self.pseudo_labels):
      return age, gen
    age_pred = np.mean([i[uuid][0] for i in self.pseudo_labels])
    gen_pred = np.mean([i[uuid][1] for i in self.pseudo_labels])
    if age < 0:
      age = age_pred
    if gen < 0:
      gen = gen_pred
    return age, gen


class AudioRead(torch.nn.Module):
  takes = ['path', 'meta']
  provides = ['signal', 'sr', 'meta']

  def __init__(self,
               random_cut: Optional[float] = 3.0,
               min_cut: float = 0.5,
               preemphasis: Optional[float] = 0.97,
               seed: int = 1):
    super(AudioRead, self).__init__()
    self.random_cut = random_cut
    self.min_cut = float(min_cut)
    self.preemphasis = preemphasis
    self.rand = np.random.RandomState(seed=seed)
    self.sr = int(SAMPLE_RATE)
    self.resampler = dict()

  def forward(self, path: str, meta: Dict[str, Any]):
    with torch.no_grad():
      duration = meta['duration']
      sr = meta['sr']
      cough = meta.get('cough_intervals', [])
      ## convert all timestamp to fraction
      if not isinstance(cough, list):
        cough = []
      else:
        cough = [dict(start=e['start'] / duration, end=e['end'] / duration,
                      labels=e['labels'])
                 for e in cough]
      ## load normally
      if duration < 1.0 or self.random_cut <= 0:
        y, sr = torchaudio.load(path)
      ## random cut utterance
      else:
        num_frames = int(self.random_cut * sr)
        if isinstance(cough, list) and len(cough) > 0:
          # random pick an event
          loop_breaker = 0
          while True:
            event = self.rand.choice(cough, 1)[0]
            if abs(duration - event['start']) >= self.min_cut:
              break
            loop_breaker += 1
            if loop_breaker >= 10:
              break
          # cut the audio
          start = int(event['start'] * duration * sr)
          end = int(np.ceil(event['end'] * duration * sr))
          offset = max(0., (num_frames - (end - start)) / 2)
          start = max(0, start - int(self.rand.rand() * offset))
          y, sr = torchaudio.load(path,
                                  frame_offset=start,
                                  num_frames=num_frames)
          start_frac = start / (duration * sr)
          duration_frac = min(1.0, max(y.shape) / sr / duration)
          cough = [dict(labels=i['labels'],
                        **map_fraction(i['start'], i['end'], duration, sr,
                                       offset=start, frames=max(y.shape)))
                   for i in cough
                   if start_frac <= i['start'] <= start_frac + duration_frac]
        else:  # no cough annotation
          offset = max(0, int(
            self.rand.rand() * (int(sr * duration) - num_frames - 1)))
          y, sr = torchaudio.load(path,
                                  frame_offset=offset,
                                  num_frames=num_frames)
      ## load audio
      assert y.shape[0] == 1, y.shape
      y = y.squeeze(0)
      ## downsampling
      if sr != self.sr:
        if sr not in self.resampler:
          self.resampler[sr] = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=self.sr)
        # some error during resampling here
        try:
          y = self.resampler[sr](y)
        except Exception as e:
          write_errors(path, str(meta), str(y.shape), str(sr), str(e))
          raise e
      sr = self.sr
      ## post processing
      if self.preemphasis > 0.:
        y = preemphasis(y, coef=self.preemphasis)
      meta['cough_intervals'] = cough
    return y, sr, meta


class Pitch(torch.nn.Module):
  takes = ['signal', 'sr']
  provides = ['pitch']

  def __init__(self):
    super(Pitch, self).__init__()

  def forward(self, signal, sr):
    return compute_kaldi_pitch(signal, sample_rate=sr,
                               frame_length=25.0,
                               frame_shift=10.0,
                               min_f0=25, max_f0=800)[:, 0]
    # return detect_pitch_frequency(
    #   signal, sr,
    #   frame_time=10 ** (-2),
    #   win_length=15,
    #   freq_low=50, freq_high=4000)


class VAD(torch.nn.Module):
  takes = ['signal']
  provides = ['vad', 'vad_y', 'energies']

  def __init__(self,
               n_fft: int = 400,
               win_length: float = 0.025,
               hop_length: float = 0.010,
               threshold: float = 35.):
    super(VAD, self).__init__()
    self.n_fft = int(n_fft)
    self.win_length = float(win_length)
    self.hop_length = float(hop_length)
    self.threshold = float(threshold)
    self.sr = SAMPLE_RATE

    fft_window = librosa.filters.get_window(
      'hann', int(win_length * SAMPLE_RATE), fftbins=True)
    fft_window = librosa.util.pad_center(fft_window, n_fft)
    fft_window = fft_window.reshape((-1, 1))
    self.fft_window = fft_window

  def energies(self, y: torch.Tensor):
    dtype = y.dtype
    y = y.numpy()
    hop_length = int(self.hop_length * self.sr)
    y = np.pad(y, int(self.n_fft // 2), mode='reflect')
    frames = librosa.util.frame(y,
                                frame_length=self.n_fft,
                                hop_length=hop_length)
    frames = (self.fft_window * frames).T
    # [n_frames, samples]
    frames = torch.tensor(frames, dtype=dtype)
    log_energy = (frames ** 2).sum(axis=1)
    log_energy = torch.log(log_energy + 1e-10)
    return log_energy, frames

  def forward(self, y: torch.Tensor):
    energies, frames = self.energies(y)
    vad = vad_threshold(frames, threshold=self.threshold)
    vad_y = vad.repeat(int(np.ceil(y.shape[0] / vad.shape[0])))
    n_diff = abs(vad_y.shape[0] - y.shape[0])
    vad_y = vad_y[n_diff // 2: (y.shape[0] + n_diff // 2)]
    return vad, vad_y, energies


class LabelEncoder(torch.nn.Module):
  takes = ['meta']
  provides = ['result', 'gender', 'age']

  def __init__(self,
               pseudo_labeling=False,
               pseudo_threshold=0.5,
               pseudo_soft=False,
               pseudo_rand=False):
    super().__init__()
    self.gender_encoder = dict(unknown=-1,
                               female=0,
                               male=1)
    self.age_encoder = dict(unknown=-1,
                            group_0_2=0,
                            group_3_5=1,
                            group_6_13=2,
                            group_14_18=3,
                            group_19_33=4,
                            group_34_48=5,
                            group_49_64=6,
                            group_65_78=7,
                            group_79_98=8)
    self.result_encoder = {'unknown': -1,
                           '0': 0,
                           '1': 1,
                           0: 0,
                           1: 1,
                           -1: -1}
    # use raw probability value instead of hard value
    self.pseudo_labeler = None
    self.pseudo_age_gen = None
    if pseudo_labeling:
      self.pseudo_age_gen = PseudoAgeGen.get_labeler()
      self.pseudo_labeler = PseudoLabeler.get_labeler(
        pseudo_soft=pseudo_soft, pseudo_rand=pseudo_rand,
        pseudo_threshold=pseudo_threshold)

  def forward(self, meta: Dict[str, Any]):
    # age, gender
    age = self.age_encoder[meta.get('subject_age', 'unknown')]
    if 0 <= age <= 4:
      age = 0  # young
    elif age >= 5:
      age = 1  # old
    gender = self.gender_encoder[meta.get('subject_gender', 'unknown')]
    # result
    result = self.result_encoder[meta.get('assessment_result', -1)]
    if self.pseudo_labeler is not None and result < 0:
      result = self.pseudo_labeler.label(meta['uuid'])
      age, gender = self.pseudo_age_gen.label(meta['uuid'], age, gender)
    return result, gender, age


class AcousticFeatures(torch.nn.Module):
  takes = ['path', 'meta']
  provides = ['signal', 'energies', 'spec', 'vad', 'length', 'status']

  def __init__(self,
               sample_rate: int = 8000,
               random_cut: Optional[float] = 2.0,
               preemphasis: Optional[float] = 0.97,
               training: bool = True,
               n_fft: int = 400,
               win_length: float = 0.025,
               hop_length: float = 0.010,
               vad_threshold: float = 30.,
               cmn_window: int = 100):
    super().__init__()
    self.sample_rate = sample_rate
    self.win_length = win_length
    self.hop_length = hop_length
    self.vad_threshold = vad_threshold
    self.n_fft = n_fft
    self.preemphasis = preemphasis
    self.random_cut = random_cut
    self.training = training

    self.fbank = torchaudio.transforms.MelSpectrogram(
      sample_rate=sample_rate,
      n_fft=n_fft,
      win_length=int(win_length * sample_rate),
      hop_length=int(hop_length * sample_rate),
      f_min=0.0,
      f_max=None,
      n_mels=80,
      power=2.0)
    self.toDB = torchaudio.transforms.AmplitudeToDB(stype='power')

    self.wcmn = torchaudio.transforms.SlidingWindowCmn(
      cmn_window=cmn_window,
      center=True,
      norm_vars=True)

    self.time_aug = TimeDomainSpecAugment(
      sample_rate=sample_rate,
      speeds=[90, 95, 100, 105, 110],
      drop_freq_count_low=0,
      drop_freq_count_high=3,
      drop_chunk_count_low=0,
      drop_chunk_count_high=5,
      drop_chunk_length_low=1000,
      drop_chunk_length_high=4000,
      drop_chunk_noise_factor=0.,
    )
    self.spec_aug = SpecAugment(time_warp=True,
                                time_warp_window=5,
                                time_warp_mode="bicubic",
                                freq_mask=True,
                                freq_mask_width=(0, 10),
                                n_freq_mask=2,
                                time_mask=True,
                                time_mask_width=(0, 30),
                                n_time_mask=2,
                                replace_with_zero=True)

  def forward(self, filepath: str, meta: Dict[str, Any]):
    with torch.no_grad():
      y, sr = torchaudio.load(filepath)
      y = y.squeeze(0)
      length = 1.0
      # preemphasis
      if self.preemphasis is not None:
        y = preemphasis(y, 0.97)
      # time augmentation
      if self.training:
        y = y.unsqueeze(0)
        y = self.time_aug(y, torch.ones(1))
        y = y.squeeze(0)
      # FBank
      spec = self.fbank(y)
      spec = self.toDB(spec)
      spec = torch.transpose(spec, 0, 1)
      # frames' energy
      energies, frames = self.energies(y)
      vad = vad_threshold(frames, threshold=self.vad_threshold)
      spec = spec[vad]
      if spec.shape[0] < 10 or torch.all(spec < 0.):
        return None, None, None, None, None, False
      # normalize
      spec = self.wcmn(spec.unsqueeze(0)).squeeze(0)
      # random cut
      if self.random_cut is not None:
        n_frames = int(self.random_cut / self.hop_length)
        if spec.shape[0] < n_frames:
          length = spec.shape[0] / n_frames
          spec = F.pad(spec, [0, 0, 0, n_frames - spec.shape[0]],
                       mode='constant', value=0.)
        elif spec.shape[0] > n_frames:
          start = random.randint(0, spec.shape[0] - n_frames - 1)
          spec = spec[start:start + n_frames]
      # spec augmentation
      if self.training:
        spec = spec.unsqueeze(0)
        aug = torch.rand(1)
        if aug <= 0.25:
          spec = self.spec_aug.time_warp(spec)
        elif aug <= 0.5:
          spec = self.spec_aug.mask_along_axis(spec, 1)
        elif aug <= 0.75:
          spec = self.spec_aug.mask_along_axis(spec, 2)
        spec = spec.squeeze(0)
      return y, energies, spec, vad, length, True
