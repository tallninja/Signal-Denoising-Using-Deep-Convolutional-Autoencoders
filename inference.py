import librosa
import pywt
import numpy as np
from constants import *
from model import model
from audio_utils import load_audio_file

weights = os.path.join(BASE_DIR, "weights", "white.pth")
optimizer = torch.optim.Adam(model.parameters())
model_state = torch.load(weights, map_location=DEVICE)
model.load_state_dict(model_state)

model.eval()


def denoise_audio_wt(audio_data, sr=SAMPLE_RATE, wavelet='db2', threshold_type='soft', threshold_value=None):
    # Perform wavelet transform
    coeffs = pywt.wavedec(audio_data, wavelet)

    # Set a threshold for coefficients
    if threshold_value is None:
        threshold_value = 0.2 * max(coeffs[0])

    # Thresholding
    coeffs_thresholded = [pywt.threshold(
        c, threshold_value, mode=threshold_type) for c in coeffs]

    # Inverse wavelet transform
    denoised = pywt.waverec(coeffs_thresholded, wavelet)

    return sr, denoised


def denoise_audio(audio_data, sr=SAMPLE_RATE):
    audio_data = torch.stft(input=audio_data, n_fft=N_FFT,
                            hop_length=HOP_LENGTH, normalized=True)
    print(audio_data.shape)
    denoised = model(audio_data[None, ...], is_istft=True)
    denoised = denoised[0].view(-1).detach().cpu().numpy()
    return sr, denoised[np.argmax(denoised > 0):]


def infer_model(audio_file):
    audio_data = load_audio_file(audio_file, MAX_LEN)
    return denoise_audio(audio_data)


def infer_model_wt(audio_file):
    audio_data, sr = librosa.load(audio_file, sr=None)
    return denoise_audio_wt(audio_data)
