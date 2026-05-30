"""
Neuro-Acoustic Grounding — pipeline principale.
"""
from src.recording.recording_data import RecordingData
from src.preprocessing.preprocessing import Preprocessing
from src.features.features import FeatureExtractor
from src.metrics.metrics import Metrics
from src.ml.predict import AnxietyPredictor
from src.audio.audio_engine import AudioEngine
import math
import time
import numpy as np

# --- Configurazione ---
ATTIVA_MODELLO = 'eegnet' # Ora usa il Deep Learning!

EEG_BUFFER_SIZE = 1024   
BAND_BUFFER_SIZE = 10    
LOW_FREQ = 1.0
HIGH_FREQ = 45.0   
USE_ML = True           
USE_SOUNDSCAPES = True 

SMOOTHING_ALPHA = 0.3 

record = RecordingData()
preprocessing = Preprocessing()
features = FeatureExtractor()
metrics = Metrics()
audio = AudioEngine(use_soundscapes=USE_SOUNDSCAPES)

predictor = AnxietyPredictor(use_ml=USE_ML, model_type=ATTIVA_MODELLO)

print(f'\n--- Sistema avviato (Modello Attivo: {ATTIVA_MODELLO.upper()}) ---')
print('Premi Ctrl+C per terminare.\n')
audio.start()

smoothed_prob = 0.5 

try:
    while True:
        record.acquisition_record_data(eeg_buffer_size=EEG_BUFFER_SIZE)
        record.create_data_structure()

        # Preprocessing rigorosamente senza CAR per asimmetrie e CNN
        preprocessing.band_pass_filter(record.raw, low=LOW_FREQ, high=HIGH_FREQ)

        features.extract_bands_power(raw=preprocessing.raw)
        metrics.extract_metrics(band_buffer=features.band_buffer, band_buffer_size=BAND_BUFFER_SIZE)
         
        # Assicurati che metrics.tbr_af7 sia presente per iniziare le previsioni
        if metrics.tbr_af7 is not None:
            raw_data = preprocessing.raw.get_data()
            
            af7_raw  = raw_data[0] 
            tp9_raw  = raw_data[1] if raw_data.shape[0] > 1 else raw_data[0]
            tp10_raw = raw_data[2] if raw_data.shape[0] > 2 else raw_data[0]
            af8_raw  = raw_data[3] if raw_data.shape[0] > 3 else raw_data[1] 
            
            _, raw_prob = predictor.predict(
                tbr_af7=metrics.tbr_af7,
                tbr_tp9=metrics.tbr_tp9,
                tbr_tp10=metrics.tbr_tp10,
                tbr_af8=metrics.tbr_af8,
                faa=metrics.faa,
                taa=metrics.taa,
                raw_af7=af7_raw,
                raw_tp9=tp9_raw,
                raw_tp10=tp10_raw,
                raw_af8=af8_raw,
                anxiety_score=metrics.anxiety_score
            )
            
            smoothed_prob = (SMOOTHING_ALPHA * raw_prob) + ((1.0 - SMOOTHING_ALPHA) * smoothed_prob)
            audio.update(anxiety_prob=smoothed_prob)

        record.resize_eeg_buffer()
        features.resize_band_buffer(band_buffer_size=BAND_BUFFER_SIZE)

except KeyboardInterrupt:
    print('\nChiusura...')
finally:
    audio.stop()