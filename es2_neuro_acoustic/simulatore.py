"""
Simulatore EEG — genera un segnale LSL sintetico per test offline.
Esegui questo script al posto di connect.py quando il Muse 2 non è disponibile.

Simula due stati alternati ogni 10 secondi: CALMA e ANSIA.
"""
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

SFREQ = 256
N_CHANNELS = 4  # TP9, AF7, AF8, TP10
CHUNK_SIZE = 32

info = StreamInfo(
    name='Muse',
    type='EEG',
    channel_count=N_CHANNELS,
    nominal_srate=SFREQ,
    channel_format='float32',
    source_id='sim_muse2',
)
outlet = StreamOutlet(info)

print('Simulatore EEG avviato. Premi Ctrl+C per terminare.')
t = 0.0

try:
    while True:
        # Alterna calma (theta alta, beta bassa) e ansia ogni 10s
        calm = (int(time.time()) % 20) < 10
        label = 'CALMA' if calm else 'ANSIA'

        chunk = []
        for _ in range(CHUNK_SIZE):
            if calm:
                # calma: theta dominante, beta bassa
                af7 = 0.6 * np.sin(2 * np.pi * 6 * t) + 0.1 * np.random.randn()
                af8 = 0.5 * np.sin(2 * np.pi * 6 * t) + 0.1 * np.random.randn()
            else:
                # ansia: beta dominante, theta bassa
                af7 = 0.2 * np.sin(2 * np.pi * 6 * t) + 0.6 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn()
                af8 = 0.2 * np.sin(2 * np.pi * 6 * t) + 0.5 * np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn()

            sample = [
                0.1 * np.random.randn(),  # TP9 (non usato)
                float(af7),               # AF7
                float(af8),               # AF8
                0.1 * np.random.randn(),  # TP10 (non usato)
            ]
            chunk.append(sample)
            t += 1.0 / SFREQ

        outlet.push_chunk(chunk)
        print(f'[SIM] Stato: {label}', end='\r')
        time.sleep(CHUNK_SIZE / SFREQ)

except KeyboardInterrupt:
    print('\nSimulatore terminato.')
