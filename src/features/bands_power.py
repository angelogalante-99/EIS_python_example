import numpy as np
from scipy.signal import welch

# Bande EEG in Hz
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
BETA_BAND = (13, 30)

SFREQ = 256
EPOCH_DURATION = 2  # secondi
NPERSEG = SFREQ * EPOCH_DURATION


class BandsPower:
    def __init__(self):
        self.band_buffer = []

    def _band_power(self, data: np.ndarray, low: float, high: float) -> float:
        freqs, psd = welch(data, fs=SFREQ, nperseg=min(NPERSEG, len(data)))
        idx = np.logical_and(freqs >= low, freqs <= high)
        return float(np.trapezoid(psd[idx], freqs[idx]))

    def extract_bands_power(self, raw, sfreq: int = SFREQ):
        data = raw.get_data()  # shape (2, samples): AF7=0, AF8=1
        af7 = data[0]
        af8 = data[1]

        epoch = {
            'AF7': {
                'theta': self._band_power(af7, *THETA_BAND),
                'alpha': self._band_power(af7, *ALPHA_BAND),
                'beta':  self._band_power(af7, *BETA_BAND),
            },
            'AF8': {
                'theta': self._band_power(af8, *THETA_BAND),
                'alpha': self._band_power(af8, *ALPHA_BAND),
                'beta':  self._band_power(af8, *BETA_BAND),
            },
        }
        self.band_buffer.append(epoch)

    def resize_band_buffer(self, band_buffer_size: int):
        if len(self.band_buffer) == band_buffer_size:
            self.band_buffer = self.band_buffer[1:]
