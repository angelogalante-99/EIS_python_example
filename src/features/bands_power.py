import numpy as np
from scipy.signal import welch

# Bande EEG in Hz
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 13)
BETA_BAND = (13, 30)

SFREQ = 256
EPOCH_DURATION = 4  # secondi
NPERSEG = SFREQ * EPOCH_DURATION

class BandsPower:
    def __init__(self):
        self.band_buffer = []

    def _band_power(self, data: np.ndarray, low: float, high: float) -> float:
        freqs, psd = welch(data, fs=SFREQ, nperseg=min(NPERSEG, len(data)))
        idx = np.logical_and(freqs >= low, freqs <= high)
        return float(np.trapz(psd[idx], freqs[idx]))

    def extract_bands_power(self, raw, sfreq: int = SFREQ):
        data = raw.get_data() 
        
        # ORDINE DATASET: 0:AF7, 1:TP9, 2:TP10, 3:AF8
        af7  = data[0]
        tp9  = data[1] if data.shape[0] > 1 else data[0]
        tp10 = data[2] if data.shape[0] > 2 else data[0]
        af8  = data[3] if data.shape[0] > 3 else data[1]

        epoch = {
            'AF7': {
                'theta': self._band_power(af7, *THETA_BAND),
                'alpha': self._band_power(af7, *ALPHA_BAND),
                'beta':  self._band_power(af7, *BETA_BAND),
            },
            'TP9': {
                'theta': self._band_power(tp9, *THETA_BAND),
                'alpha': self._band_power(tp9, *ALPHA_BAND),
                'beta':  self._band_power(tp9, *BETA_BAND),
            },
            'TP10': {
                'theta': self._band_power(tp10, *THETA_BAND),
                'alpha': self._band_power(tp10, *ALPHA_BAND),
                'beta':  self._band_power(tp10, *BETA_BAND),
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