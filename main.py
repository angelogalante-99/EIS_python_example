"""
Neuro-Acoustic Grounding — pipeline principale.

Flusso:
  Muse 2 → LSL → Preprocessing → [Modello Universale] → Predizione → Smoothing → Audio Engine
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
ATTIVA_MODELLO = 'minirocket'

EEG_BUFFER_SIZE = 1024   
BAND_BUFFER_SIZE = 10    
LOW_FREQ = 1.0
HIGH_FREQ = 40.0
USE_ML = True           
USE_SOUNDSCAPES = True 

# --- Configurazione Smoothing Audio (EMA) ---
# Alpha regola la velocità di reazione (da 0.0 a 1.0).
# 1.0 = Nessuno smoothing (salto istantaneo)
# 0.2 = Molto morbido (ci mette qualche secondo a raggiungere il picco)
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

# Variabile per ricordare la probabilità precedente
smoothed_prob = 0.5 

try:
    while True:
        # 1. Acquisizione Dati
        record.acquisition_record_data(eeg_buffer_size=EEG_BUFFER_SIZE)
        record.create_data_structure()

        # 2. Preprocessing
        preprocessing.band_pass_filter(record.raw, low=LOW_FREQ, high=HIGH_FREQ)

        # 3. Estrazione Feature (TBR/FAA)
        features.extract_bands_power(raw=preprocessing.raw)
        metrics.extract_metrics(band_buffer=features.band_buffer, band_buffer_size=BAND_BUFFER_SIZE)
         
        # 4. Predizione Dinamica
       # 4. Predizione Dinamica
        if metrics.tbr is not None:
            # Estraiamo i dati GIA' FILTRATI in-place da MNE
            raw_data = preprocessing.raw.get_data()
            af7_raw = raw_data[0] 
            af8_raw = raw_data[3] if raw_data.shape[0] > 3 else raw_data[1] 
            
            _, raw_prob = predictor.predict(
                tbr=metrics.tbr,
                faa=metrics.faa,
                raw_af7=af7_raw,
                raw_af8=af8_raw,
                anxiety_score=metrics.anxiety_score
            )
            
            # --- 5. SMOOTHING DELLA PROBABILITÀ ---
            smoothed_prob = (SMOOTHING_ALPHA * raw_prob) + ((1.0 - SMOOTHING_ALPHA) * smoothed_prob)
 
            audio.update(anxiety_prob=smoothed_prob)

        # Pulizia buffer
        record.resize_eeg_buffer()
        features.resize_band_buffer(band_buffer_size=BAND_BUFFER_SIZE)

except KeyboardInterrupt:
    print('\nChiusura...')
finally:
    audio.stop()