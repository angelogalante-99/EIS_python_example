from src.utils.preprocessing import signal_filter, notch_filter
from src.utils.create_epochs import create_epochs
import numpy as np


class Filter:
    def __init__(self):
        self.epoch_filtered_to_emotion = None

    def filter_eeg_to_emotion_recognition(self, eeg_buffer, low_freq, high_freq):
        epochs = np.array(eeg_buffer)[:, 0:4].T
        epochs = create_epochs(data=epochs, sfreq=256)
        epochs = epochs.filter(low_freq, high_freq).get_data()
        self.epoch_filtered_to_emotion = epochs
