import numpy as np
from pylsl import StreamInlet, resolve_stream
from src.utils.create_raw_complete import create_raw_complete

class RecordingData:
    def __init__(self):
        self.raw = None
        self.streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(self.streams[0])
        self.eeg_buffer = []
        self.EEG = None
        self.sfreq = 256

    def acquisition_record_data(self, eeg_buffer_size):
        self.inlet.flush()
        while len(self.eeg_buffer) < eeg_buffer_size:
            sample, timestamp = self.inlet.pull_sample()
            self.eeg_buffer.append(sample)

    def create_data_structure(self):
        self.EEG = np.array(self.eeg_buffer)[:, 0:4].T
        self.raw = create_raw_complete(self.EEG, sfreq=256)

    def resize_eeg_buffer(self):
        self.eeg_buffer = self.eeg_buffer[1:]
