import numpy as np
from scipy import interpolate
import torchaudio
import torch
import librosa

from constants import SAMPLE_RATE


def resample(original, old_rate, new_rate):
    if new_rate != old_rate:
        return librosa.resample(original, orig_sr=old_rate, target_sr=new_rate)
    return original


def load_audio_file(file_path='sample.wav', max_len=165000):
    audio_data, sr = torchaudio.load(file_path)
    audio_data = resample(audio_data, sr, SAMPLE_RATE)
    audio_data = audio_data.numpy()
    current_len = audio_data.shape[1]
    output = np.zeros((1, max_len), dtype='float32')
    output[0, -current_len:] = audio_data[0, :max_len]
    return torch.from_numpy(output)


def adjust_audio_length(audio_data, max_len=165000):
    adjusted_audio = librosa.util.fix_length(audio_data, size=max_len)
    adjusted_audio = adjusted_audio.astype(np.float32)
    return torch.from_numpy(adjusted_audio)


def save_audio_file(dest_path, audio_data, sample_rate=48000, bit_precision=16):
    audio_data = np.reshape(audio_data, (1, -1))
    audio_data = torch.from_numpy(audio_data)
    torchaudio.save(dest_path, audio_data, sample_rate,
                    bits_per_sample=bit_precision)
