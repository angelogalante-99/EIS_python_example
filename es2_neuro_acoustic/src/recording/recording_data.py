import numpy as np
from pylsl import StreamInlet, resolve_stream
from src.utils.create_raw_complete import create_raw_complete

# Indici dei canali AF7 e AF8 nello stream Muse 2 (0=TP9, 1=AF7, 2=AF8, 3=TP10)
AF7_IDX = 1
AF8_IDX = 2


class RecordingData:
    def __init__(self):
        self.raw = None
        self.streams = resolve_stream('type', 'EEG')
        self.inlet = StreamInlet(self.streams[0])
        self.eeg_buffer = []
        self.EEG = None
        self.sfreq = 256

    def acquisition_record_data(self, eeg_buffer_size: int):
        self.inlet.flush()
        while len(self.eeg_buffer) < eeg_buffer_size:
            sample, _ = self.inlet.pull_sample()
            self.eeg_buffer.append(sample)

    def create_data_structure(self):
        arr = np.array(self.eeg_buffer)
        # Estrae solo AF7 e AF8 (elettrodi prefrontali)
        self.EEG = arr[:, [AF7_IDX, AF8_IDX]].T
        self.raw = create_raw_complete(self.EEG, sfreq=self.sfreq)

    def resize_eeg_buffer(self):
        self.eeg_buffer = self.eeg_buffer[1:]
