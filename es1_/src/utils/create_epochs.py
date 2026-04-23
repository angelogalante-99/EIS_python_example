import mne
import numpy as np


def create_epochs(data, sfreq):
    data = data * 1e-6
    data = np.expand_dims(data, axis=0)
    ch_names = ['AF3', 'TP9', 'TP10', 'AF4']
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
    epochs = mne.EpochsArray(data, info)
    epochs.set_montage('standard_1020')

    return epochs


# def create_epochs_2_channels(data, sfreq):
#     data = data * 1e-6
#     ch_names = ['AF3', 'AF4']
#     ch_types = ['eeg'] * len(ch_names)
#     info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
#     epochs = mne.EpochsArray(data, info)
#     epochs.set_montage('standard_1020')
#
#     return epochs
