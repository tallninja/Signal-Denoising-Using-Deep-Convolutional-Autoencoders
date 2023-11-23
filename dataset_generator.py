import os
import random
import numpy as np
import torchaudio
from pydub import AudioSegment
from tqdm import tqdm

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


def diff_noise_type(files, noise_type):
    result = []
    for file_name in files:
        if file_name.endswith(".wav"):
            fname = file_name.split("-")
            if fname[1] != str(noise_type):
                result.append(file_name)
    return result


def same_noise_type(files, noise_type):
    result = []
    for file_name in files:
        if file_name.endswith(".wav"):
            fname = file_name.split("-")
            if fname[1] == str(noise_type):
                result.append(file_name)
    return result


def corrupt_audio_file(audio_file, noise_file, snr):
    original_audio, _ = torchaudio.load(audio_file)
    original_audio = original_audio.numpy()
    original_audio = np.reshape(original_audio, -1)
    # create power array of the original audio
    original_audio_power = original_audio ** 2
    # calculate average signal power of the original audio and convert to dB
    original_audio_avg_power = np.mean(original_audio_power)
    original_audio_avg_power_db = 10 * np.log10(original_audio_avg_power)
    # calculate noise power
    added_noise_avg_power_db = original_audio_avg_power_db - snr

    noise, _ = torchaudio.load(noise_file)
    noise = noise.numpy()
    noise = np.reshape(noise, -1)
    noise_power = noise ** 2
    noise_avg_power = np.mean(noise_power)
    noise_avg_power_db = 10 * np.log10(noise_avg_power)

    delta_noise_power = added_noise_avg_power_db - noise_avg_power_db

    try:
        source_audio = AudioSegment.from_file(audio_file)
        noise_audio = AudioSegment.from_file(noise_file)
    except:
        pass

    noise_audio = noise_audio + delta_noise_power
    corrupted_audio = source_audio.overlay(noise_audio, times=5)
    # corrupted_audio.export(dest, format='wav')
    return corrupted_audio


def corrupt_audio_file_with_noise_type(filename, target_folder, dest, snr, noise_type, gen_noise_type):
    success = False
    fold_names = get_fold_names()

    while not success:
        try:
            fold = np.random.choice(fold_names, 1, replace=False)[0]
            fold_dir = os.path.join(URBAN_SOUND_8K_DIR, fold)
            fold_noises = os.listdir(fold_dir)
            possible_noises = gen_noise_type(fold_noises, noise_type)
            possible_noises_count = len(possible_noises)
            choice = np.random.choice(
                possible_noises_count, 1, replace=False)[0]
            noise_file = possible_noises[choice]

            noise_file = os.path.join(fold_dir, noise_file)
            audio_file = os.path.join(target_folder, filename)
            dest_path = os.path.join(dest, filename)

            corrupted_audio = corrupt_audio_file(audio_file, noise_file, snr)
            corrupted_audio.export(dest_path, format='wav')
            success = True
        except Exception as e:
            pass


def generate_train_data(noise_type):
    input_folder = os.path.join(
        DATASETS_FOLDER, f'class_{noise_type}_train_input')
    output_folder = os.path.join(
        DATASETS_FOLDER, f'class_{noise_type}_train_output')

    if not os.path.exists(input_folder):
        print("Creating train input folder...")
        os.makedirs(input_folder)

    if not os.path.exists(output_folder):
        print("Creating train output folder...")
        os.makedirs(output_folder)

    for file_ in tqdm(os.listdir(TARGET_FOLDER)):
        filename = os.fsdecode(file_)
        if filename.endswith(".wav"):
            snr = random.randint(0, 10)
            corrupt_audio_file_with_noise_type(filename, TARGET_FOLDER, input_folder, snr,
                                               noise_type, same_noise_type)
            corrupt_audio_file_with_noise_type(filename, output_folder, snr,
                                               noise_type, diff_noise_type)


def generate_test_data(noise_type):
    target_folder = os.path.join(DATASETS_FOLDER, "clean_testset_wav")
    input_folder = os.path.join(
        DATASETS_FOLDER, f'class_{noise_type}_test_input')

    if not os.path.exists(input_folder):
        print("Making test input folder")
        os.makedirs(input_folder)

    for file_ in tqdm(os.listdir(target_folder)):
        filename = os.fsdecode(file_)
        if filename.endswith(".wav"):
            snr = random.randint(0, 10)
            corrupt_audio_file_with_noise_type(
                filename, target_folder, input_folder, snr, noise_type, same_noise_type)


def generate_dataset():
    for key in noise_classes:
        print("\t{} : {}".format(key, noise_classes[key]))
    noise_type = int(input("Enter the noise class dataset to generate :\t"))

    # print("##################### GENERATING TRAIN DATA #####################")
    # generate_train_data(noise_type)
    print("##################### GENERATING TEST DATA #####################")
    generate_test_data(noise_type)


if __name__ == '__main__':
    generate_dataset()
