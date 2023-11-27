import os
import torch

SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000
MAX_LEN = 165000

# Check GPU is available
GPU_AVAILABLE = torch.cuda.is_available()

if (GPU_AVAILABLE):
    print('Using GPU.')
else:
    print('No GPU available, using CPU.')

DEVICE = torch.device('cuda' if GPU_AVAILABLE else 'cpu')

BASE_DIR = os.path.curdir
