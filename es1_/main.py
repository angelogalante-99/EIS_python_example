from src.recording.recording import RecordingData
from src.preprocessing.preprocessing import Preprocessing
from src.features.features import FeatureExtractor
from src.metrics.metrics import Metrics
from src.send_data.send_data import SendData

record = RecordingData()
preprocessing = Preprocessing()
features = FeatureExtractor()
metrics = Metrics()
send_data = SendData()


band_buffer_size = 10
notch_frequency = 50
sfreq = 256
low_freq = 1
high_freq = 30

while True:
    record.acquisition_record_data(eeg_buffer_size=1281)
    # create data for mental states
    record.create_data_structure()
    preprocessing.band_pass_filter(record.raw, low=low_freq, high=high_freq)
    features.extract_bands_power(raw=preprocessing.raw, sfreq=sfreq)
    metrics.extract_metrics(band_buffer=features.band_buffer, band_buffer_size=band_buffer_size)
    send_data.send_osc_data(focus=metrics.focus, relaxation=metrics.relaxation,
                            stress=metrics.stress, engagements=metrics.engagement, valence=metrics.valence)
    record.resize_eeg_buffer()
    features.resize_band_buffer(band_buffer_size=band_buffer_size)
