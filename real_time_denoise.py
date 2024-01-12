#!/usr/bin/env python3

import time
import librosa
import sounddevice as sd
import soundfile as sf
import numpy as np
from inference import denoise_audio
import torch

from constants import MAX_LEN, SAMPLE_RATE


def record_audio(duration, samplerate=SAMPLE_RATE):
    print("Recording...")
    audio_data = sd.rec(int(samplerate * duration),
                        samplerate=samplerate, channels=1)
    sd.wait()
    print("Recording done.")
    return audio_data


def split_audio_into_clips(audio_data, clip_duration):
    print(f"Audio data length: {audio_data.shape[0]}")  # 576000
    clip_length = SAMPLE_RATE * clip_duration
    clips = [audio_data[i:i+clip_length]
             for i in range(0, len(audio_data), clip_length)]
    return clips


def combine_audio_arrays(audio_arrays):
    print("Assembling audio...")
    # Combine the audio arrays into a single array
    combined_audio = np.concatenate(audio_arrays)
    return combined_audio


def play_audio(audio_data, samplerate=SAMPLE_RATE):
    print("Playing audio...")
    sd.play(audio_data, samplerate)
    sd.wait()
    print("Audio playback done.")


if __name__ == "__main__":
    recording_duration = 12  # in seconds
    clip_duration = 3  # in seconds

    audio_data = record_audio(recording_duration, SAMPLE_RATE)
    sf.write("input.wav", audio_data, samplerate=SAMPLE_RATE,
             format='wav', subtype='PCM_24')

    audio_clips = split_audio_into_clips(audio_data, clip_duration)
    print(f"No of audio clips: {len(audio_clips)}")

    # Convert clips to numpy arrays, apply STFT, denoise with the model, and add them to a list
    denoised_audios_array = []
    for i, clip in enumerate(audio_clips):
        print(f"clip #{i+1}")
        clip = clip.reshape((1, 144000))
        current_len = clip.shape[1]
        clip_np = np.zeros((1, MAX_LEN), dtype='float32')
        clip_np[0, -current_len:] = clip[0, :MAX_LEN]
        audio_tensor = torch.from_numpy(clip_np)
        sr, denoised_result = denoise_audio(audio_tensor)
        denoised_audios_array.append(denoised_result)

    combined_audio = combine_audio_arrays(denoised_audios_array)
    sf.write("output.wav", combined_audio, samplerate=SAMPLE_RATE,
             format='wav', subtype='PCM_24')
    play_audio(combined_audio, sr)
