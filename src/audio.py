# https://github.com/keithito/tacotron/blob/master/util/audio.py
# https://github.com/carpedm20/multi-speaker-tacotron-tensorflow/blob/master/audio/__init__.py
# I only changed the hparams to usual parameters from original code.

import numpy as np
from scipy import signal
import librosa

def preemphasis(x, preemphasis = 0.97):
    return signal.lfilter([1, -preemphasis], [1], x)

def spectrogram(y, frame_shift_ms, frame_length_ms, sample_rate, ref_level_db=20, num_freq=256):
    D = _stft(preemphasis(y), frame_shift_ms=frame_shift_ms, frame_length_ms=frame_length_ms, sample_rate=sample_rate, num_freq=num_freq)
    S = _amp_to_db(np.abs(D)) - ref_level_db
    return _normalize(S)

def _stft(y, frame_shift_ms, frame_length_ms, sample_rate, num_freq):
    n_fft, hop_length, win_length = _stft_parameters(frame_shift_ms=frame_shift_ms, frame_length_ms=frame_length_ms, sample_rate=sample_rate, num_freq=num_freq)
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _stft_parameters(frame_shift_ms, frame_length_ms, sample_rate, num_freq):
    n_fft = (num_freq - 1) * 2
    hop_length = int(frame_shift_ms / 1000 * sample_rate)
    win_length = int(frame_length_ms / 1000 * sample_rate)
    return n_fft, hop_length, win_length
 
def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))
 
def _normalize(S, min_level_db=-100):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)
 