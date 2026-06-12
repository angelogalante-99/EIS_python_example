"""
Streaming — connessione LSL e buffering in epoche
------------------------------------------------------
Modulo "puro" I/O: si connette a uno stream LSL e accumula i campioni
in epoche da 4s (1024 campioni @ 256Hz, ordine TP9/AF7/AF8/TP10).
Non sa nulla di preprocessing, feature o modelli.

Funzioni:
  connect_lsl(...)        -> StreamInlet
  epoch_generator(inlet, stop_event=None)
      -> generator che produce un array numpy (1024, 4) per ogni
         epoca completa (epoche consecutive, nessun overlap)
"""

from pylsl import StreamInlet, resolve_byprop
import numpy as np

SFREQ     = 256.0
EPOCH_SEC = 4.0
EPOCH_LEN = int(SFREQ * EPOCH_SEC)   # 1024 campioni

CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]


def connect_lsl(stream_name=None, timeout=10.0):
    """Si connette al primo stream EEG disponibile (o a uno specifico nome)."""
    print(f"\nRicerca stream LSL EEG (timeout {timeout}s)...")
    if stream_name:
        streams = resolve_byprop("name", stream_name, timeout=timeout)
    else:
        streams = resolve_byprop("type", "EEG", timeout=timeout)

    if not streams:
        raise RuntimeError(
            "Nessuno stream LSL trovato. Avvia prima replay_real.py "
            "(o collega il Muse).")

    inlet = StreamInlet(streams[0])
    info  = inlet.info()
    print(f"  Connesso a: '{info.name()}'  "
          f"({info.channel_count()} canali, {info.nominal_srate():.0f} Hz)")
    return inlet


def epoch_generator(inlet, stop_event=None):
    """
    Generator: accumula campioni dallo stream e restituisce un'epoca
    completa (array EPOCH_LEN x 4, colonne TP9/AF7/AF8/TP10) ogni
    volta che il buffer si riempie. Epoche consecutive, nessun overlap.

    Uso:
        for epoch in epoch_generator(inlet, stop_event):
            ... fai qualcosa con epoch ...
    """
    buffer = []
    while stop_event is None or not stop_event.is_set():
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue

        buffer.append(sample[:4])  # TP9, AF7, AF8, TP10

        if len(buffer) >= EPOCH_LEN:
            epoch  = np.array(buffer[:EPOCH_LEN], dtype=float)
            buffer = buffer[EPOCH_LEN:]
            yield epoch