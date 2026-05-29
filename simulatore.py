"""
Simulatore EEG Biomedico (Calibrato) — Segnale LSL in scala microVolt (µV).
Allineato alla mappatura canali del dataset: [AF7, TP9, TP10, AF8]
"""
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet

SFREQ = 256
N_CHANNELS = 4  
CHUNK_SIZE = 32

def _generate_pink_noise(n_samples, scale_uv):
    white = np.random.randn(n_samples)
    fft = np.fft.rfft(white)
    frequencies = np.fft.rfftfreq(n_samples)
    frequencies[0] = 1.0  
    pink_fft = fft / np.sqrt(frequencies)
    return np.fft.irfft(pink_fft, n_samples) * scale_uv

info = StreamInfo('Muse', 'EEG', N_CHANNELS, SFREQ, 'float32', 'sim_muse2')
outlet = StreamOutlet(info)

print('Simulatore EEG Calibrato avviato. Premi Ctrl+C per terminare.')

try:
    while True:
        calma = (int(time.time()) % 20) < 10
        chunk = []
        
        base_noise = _generate_pink_noise(CHUNK_SIZE, 5.0) 
        t_array = np.linspace(0, CHUNK_SIZE/SFREQ, CHUNK_SIZE, endpoint=False)
        
        if calma:
            theta = 15.0 * np.sin(2 * np.pi * 6.5 * t_array)
            alpha = 30.0 * np.sin(2 * np.pi * 10.0 * t_array)
            beta  = 5.0  * np.sin(2 * np.pi * 22.0 * t_array)
            
            af7 = theta + alpha + beta + base_noise
            af8 = theta + alpha + beta + base_noise + np.random.randn(CHUNK_SIZE)*1.0 
            
        else:
            theta = 5.0  * np.sin(2 * np.pi * 6.5 * t_array)
            beta  = 25.0 * np.sin(2 * np.pi * 25.0 * t_array) 
            
            # Ansia = Meno Alpha a sinistra (AF7), più Alpha a destra (AF8)
            alpha_sx = 10.0 * np.sin(2 * np.pi * 10.0 * t_array) 
            alpha_dx = 30.0 * np.sin(2 * np.pi * 10.0 * t_array) 
            
            af7 = theta + alpha_sx + beta + base_noise
            af8 = theta + alpha_dx + beta + base_noise

        tp9_tp10 = base_noise 
        
        for i in range(CHUNK_SIZE):
            # IL FIX E' QUI: Ordine [AF7 (0), TP9 (1), TP10 (2), AF8 (3)] come da README
            chunk.append([af7[i], tp9_tp10[i], tp9_tp10[i], af8[i]])

        outlet.push_chunk(chunk)
        print(f"[SIMULATORE] Invio stato: {'CALMA' if calma else 'ANSIA'}...", end='\r')
        time.sleep(CHUNK_SIZE / SFREQ)

except KeyboardInterrupt:
    print('\nSimulatore terminato.')