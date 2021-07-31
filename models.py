from random import random
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from speechbrain.lobes.augment import TimeDomainSpecAugment, SpecAugment
from torch.nn import functional as F


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


class AcousticFeatures(torch.nn.Module):

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

    fft_window = librosa.filters.get_window(
      'hann', int(win_length * sample_rate), fftbins=True)
    fft_window = librosa.util.pad_center(fft_window, n_fft)
    fft_window = fft_window.reshape((-1, 1))
    self.fft_window = fft_window

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

  def energies(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = y.dtype
    y = y.numpy()
    hop_length = int(self.hop_length * self.sample_rate)
    y = np.pad(y, int(self.n_fft // 2), mode='reflect')
    frames = librosa.util.frame(y, frame_length=self.n_fft,
                                hop_length=hop_length)
    frames = (self.fft_window * frames).T
    # [n_frames, samples]
    frames = torch.tensor(frames, dtype=dtype)
    log_energy = (frames ** 2).sum(axis=1)
    log_energy = torch.log(log_energy + 1e-10)
    return log_energy, frames

  def forward(self, filepath: str):
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
