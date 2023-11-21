import numpy as np
from scipy import interpolate


def resample(original, old_rate, new_rate):
    if new_rate != old_rate:
        duration = original.shape[0] / old_rate
        old_time = np.linspace(0, duration, original.shape[0])
        new_time = np.linspace(0, duration, int(
            original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(old_time, original.T)
        return interpolator(new_time).T
    return original
