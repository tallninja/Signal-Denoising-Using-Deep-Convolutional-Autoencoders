import torchaudio
from constants import *
from model import model
from audio_utils import load_audio_file

weights = os.path.join(BASE_DIR, "weights", "white.pth")
optimizer = torch.optim.Adam(model.parameters())
model_state = torch.load(weights, map_location=DEVICE)
model.load_state_dict(model_state)

model.eval()


def denoise_audio(audio_data, sr=SAMPLE_RATE):
    audio_data = torch.stft(input=audio_data, n_fft=N_FFT,
                            hop_length=HOP_LENGTH, normalized=True)
    print(audio_data.shape)
    denoised = model(audio_data[None, ...], is_istft=True)
    denoised = denoised[0].view(-1).detach().cpu().numpy()
    return sr, denoised


def infer_model(audio_file):
    audio_data = load_audio_file(audio_file, MAX_LEN)
    return denoise_audio(audio_data)
