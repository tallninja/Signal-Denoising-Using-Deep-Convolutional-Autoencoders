import numpy as np
from scipy import interpolate
import torchaudio
import torch


def resample(original, old_rate, new_rate):
    if new_rate != old_rate:
        duration = original.shape[0] / old_rate
        old_time = np.linspace(0, duration, original.shape[0])
        new_time = np.linspace(0, duration, int(
            original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(old_time, original.T)
        return interpolator(new_time).T
    return original


def load_audio_file(file_path='sample.wav', max_len=165000):
    waveform, _ = torchaudio.load(file_path)
    waveform = waveform.numpy()
    current_len = waveform.shape[1]
    output = np.zeros((1, max_len), dtype='float32')
    output[0, -current_len:] = waveform[0, :max_len]
    return torch.from_numpy(output)


def save_audio_file(dest_path, audio_data, sample_rate=48000, bit_precision=16):
    audio_data = np.reshape(audio_data, (1, -1))
    audio_data = torch.from_numpy(audio_data)
    torchaudio.save(dest_path, audio_data, sample_rate,
                    bits_per_sample=bit_precision)
