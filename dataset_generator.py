import os
import random
import numpy as np
from scipy.io import wavfile
import torchaudio
from pydub import AudioSegment
from audio_utils import resample

np.random.seed(999)

BASE_DIR = os.path.curdir
DATASETS_FOLDER = os.path.join(BASE_DIR, "datasets")
URBAN_SOUND_8K_DIR = os.path.join(DATASETS_FOLDER, "UrbanSound8K", "audio")
TARGET_FOLDER = os.path.join(DATASETS_FOLDER, "clean_trainset_28spk_wav")

noise_classes = {
    0: "air_conditioner",
    1: "car_horn",
    2: "children_playing",
    3: "dog_bark",
    4: "drilling",
    5: "engine_idling",
    6: "gun_shot",
    7: "jackhammer",
    8: "siren",
    9: "street_music"
}


def get_fold_names():
    fold_names = []
    for i in range(1, 11):
        fold_names.append(f"fold{i}")
    return fold_names


def diffNoiseType(files, noise_type):
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] != str(noise_type):
                result.append(i)
    return result


def oneNoiseType(files, noise_type):
    result = []
    for i in files:
        if i.endswith(".wav"):
            fname = i.split("-")
            if fname[1] == str(noise_type):
                result.append(i)
    return result


def genNoise(filename, num_per_fold, dest):
    source_file = os.path.join(TARGET_FOLDER, filename)
    source_audio = AudioSegment.from_file(source_file)
    fold_names = get_fold_names()
    counter = 0

    for fold in fold_names:
        dirname = os.path.join(URBAN_SOUND_8K_DIR, fold)
        dirlist = os.listdir(dirname)
        total_noise = len(dirlist)
        samples = np.random.choice(total_noise, num_per_fold, replace=False)
        for sample in samples:
            noise_file = os.path.join(dirlist, dirlist[sample])
            try:
                noise_audio = AudioSegment.from_file(noise_file)
                combined = source_audio.overlay(noise_audio, times=5)
                target_dest = dest+"/" + \
                    filename[:len(filename)-4]+"_noise_"+str(counter)+".wav"
                combined.export(target_dest, format="wav")
                counter += 1
            except:
                print("Some kind of audio decoding error occurred, skipping this case")
