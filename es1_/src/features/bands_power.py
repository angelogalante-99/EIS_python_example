import neurokit2 as nk


class BandsPower:
    def __init__(self):
        self.band_buffer = []

    def extract_bands_power(self, raw, sfreq):
        by_channel = nk.eeg_power(raw, sampling_rate=sfreq, frequency_band=['Beta', 'Alpha', 'Theta', 'Delta'])
        self.band_buffer.append(by_channel)

    def resize_band_buffer(self, band_buffer_size):
        if len(self.band_buffer) == band_buffer_size:
            self.band_buffer = self.band_buffer[1:]