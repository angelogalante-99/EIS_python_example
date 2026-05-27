"""
Neuro-Acoustic Grounding — pipeline principale.

Flusso:
  Muse 2 → LSL → Preprocessing → Feature Extraction → Metrics (TBR/FAA)
  → SVM Prediction → Audio Engine (binaural beats + IADS)

Prerequisiti:
  1. connect.py in esecuzione (stream LSL attivo)
  2. Modello SVM addestrato: src/ml/train.py (richiede DREAMER)
     Oppure: USE_ML = False per usare le soglie deterministiche
"""
from src.recording.recording_data import RecordingData
from src.preprocessing.preprocessing import Preprocessing
from src.features.features import FeatureExtractor
from src.metrics.metrics import Metrics
from src.ml.predict import AnxietyPredictor
from src.audio.audio_engine import AudioEngine

# --- Configurazione ---
EEG_BUFFER_SIZE = 1024    # ~2 secondi a 256 Hz
BAND_BUFFER_SIZE = 10    # finestre da mediare
LOW_FREQ = 1.0
HIGH_FREQ = 40.0
USE_ML = True           # True se il modello SVM è già addestrato
USE_BINAURAL = False
USE_IADS = True

record = RecordingData()
preprocessing = Preprocessing()
features = FeatureExtractor()
metrics = Metrics()
predictor = AnxietyPredictor(use_ml=USE_ML)
audio = AudioEngine(use_binaural=USE_BINAURAL, use_iads=USE_IADS)

print('Sistema avviato. Premi Ctrl+C per terminare.')
audio.start()

try:
    while True:
        record.acquisition_record_data(eeg_buffer_size=EEG_BUFFER_SIZE)
        record.create_data_structure()

        preprocessing.band_pass_filter(record.raw, low=LOW_FREQ, high=HIGH_FREQ)

        features.extract_bands_power(raw=preprocessing.raw)

        metrics.extract_metrics(band_buffer=features.band_buffer, band_buffer_size=BAND_BUFFER_SIZE)

        if metrics.tbr is not None:
            _, anxiety_prob = predictor.predict(
                tbr=metrics.tbr,
                faa=metrics.faa,
                anxiety_score=metrics.anxiety_score,
            )
            audio.update(anxiety_prob=anxiety_prob)

        record.resize_eeg_buffer()
        features.resize_band_buffer(band_buffer_size=BAND_BUFFER_SIZE)

except KeyboardInterrupt:
    print('\nChiusura...')
finally:
    audio.stop()
