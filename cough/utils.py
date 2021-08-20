from random import random
from typing import Union, Tuple
import matplotlib as mpl

import librosa
import numpy as np
import torchaudio
from matplotlib import pyplot as plt
from speechbrain.lobes.features import MFCC, Fbank
from speechbrain.dataio.dataset import DynamicItemDataset

# ===========================================================================
# Helpers
# ===========================================================================
from tqdm import tqdm


def plot_spec(s, ax: Union[plt.Axes, Tuple[int, int, int]] = None):
  if isinstance(ax, (tuple, list)):
    ax = plt.subplot(*ax)
  elif ax is None:
    ax = plt.gca()
  if s.shape[0] == 1:
    s = s.squeeze(0)
  if hasattr(s, 'numpy'):
    s = s.numpy()
  ax.imshow(s, aspect='auto', origin='lower')


def save_allfig(path: str = '/tmp/tmp.pdf'):
  """Save all figures to pdf or png file"""
  figs = [plt.figure(n) for n in plt.get_fignums()]
  # ====== saving PDF file ====== #
  from matplotlib.backends.backend_pdf import PdfPages
  pp = PdfPages(path)
  for fig in figs:
    fig.savefig(pp, format='pdf', bbox_inches="tight")
  pp.close()
  plt.close('all')


def compare_fbank(y, path=None, result=None, save_path=None):
  f = Fbank(deltas=False,
            context=False,
            requires_grad=False,
            sample_rate=8000,
            f_min=0.,
            f_max=None,
            n_fft=400,
            n_mels=40,
            filter_shape="triangular",
            param_change_factor=1.0,
            param_rand_factor=0.0,
            left_frames=5,
            right_frames=5,
            win_length=25,
            hop_length=10)
  s = f(y.unsqueeze(0)).squeeze(0)
  s1 = librosa.feature.melspectrogram(y.numpy(), 8000, n_fft=400,
                                      hop_length=int(0.01 * 8000),
                                      win_length=int(0.025 * 8000),
                                      power=2.0)
  s1 = librosa.amplitude_to_db(s1)
  f2 = torchaudio.transforms.MelSpectrogram(sample_rate=8000,
                                            n_fft=400,
                                            win_length=int(0.025 * 8000),
                                            hop_length=int(0.01 * 8000),
                                            f_min=0.,
                                            f_max=None,
                                            n_mels=40,
                                            power=2.0)
  s2 = f2(y.unsqueeze(0)).squeeze(0)
  s2 = librosa.amplitude_to_db(s2.numpy())
  ######
  plt.figure()
  plt.subplot(4, 1, 1)
  plt.plot(y)
  plt.title(f"{path} - {result}", fontsize=6)
  plt.subplot(4, 1, 2)
  img = plt.imshow(s.numpy().T, aspect='auto', origin='lower')
  plt.colorbar(img)
  plt.axis('off')
  plt.subplot(4, 1, 3)
  img = plt.imshow(s1, aspect='auto', origin='lower')
  plt.colorbar(img)
  plt.axis('off')
  plt.subplot(4, 1, 4)
  img = plt.imshow(s2, aspect='auto', origin='lower')
  plt.colorbar(img)
  plt.axis('off')
  plt.tight_layout()
  if save_path is not None:
    plt.gcf().savefig(save_path)


def plot_wave(ds: DynamicItemDataset):
  ds.set_output_keys(['signal', 'meta', 'sr'])
  plt.figure(figsize=(10, 20))
  for i, j in enumerate(np.random.choice(len(ds), 10)):
    ax = plt.subplot(10, 1, i + 1)
    x = ds[int(j)]
    ax.plot(x['signal'].numpy())
    cough = x['meta']['cough_intervals']
    if isinstance(cough, list):
      for y in cough:
        start = int(y['start'] * x['signal'].shape[0])
        end = int(y['end'] * x['signal'].shape[0])
        ax.add_patch(mpl.patches.Rectangle((start, -1), end - start, 2,
                                           fill=True,
                                           color="red",
                                           alpha=0.3,
                                           linewidth=1))
    plt.title(
      f"{x['meta']['assessment_result']}-{x['meta']['cough_intervals']}",
      fontsize=6)
  plt.tight_layout()


def plot_check(ds: DynamicItemDataset):
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
