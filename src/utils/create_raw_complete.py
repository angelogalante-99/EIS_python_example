import mne
import numpy as np

CH_NAMES = ['AF7', 'AF8']
CH_TYPES = ['eeg', 'eeg']
SFREQ = 256


def create_raw_complete(eeg_data: np.ndarray, sfreq: int = SFREQ) -> mne.io.RawArray:
    info = mne.create_info(ch_names=CH_NAMES, sfreq=sfreq, ch_types=CH_TYPES)
    raw = mne.io.RawArray(eeg_data, info, verbose=False)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage, verbose=False)
    return raw
