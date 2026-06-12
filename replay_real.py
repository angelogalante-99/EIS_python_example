"""
Replay LSL — epoche reali alternate (neutral / concentrating)
------------------------------------------------------------------
Per la demo: pubblica su LSL un'epoca REALE neutral per 10s, poi
un'epoca REALE concentrating per 10s, alternando. Le due epoche sono
prese da subjectb (eeg_dataset_preprocessed.csv) e sono già validate
dalla LOSO (vedi train.py) — i 3 modelli le classificano correttamente
con alta confidenza (SVM 2%/94%, LDA 1%/100%, LGBM 18%/82%).

Ogni epoca dura 4s (1024 campioni @ 256Hz); per riempire i blocchi da
10s ogni epoca viene ripetuta ~2.5 volte. prediction.py la vedrà come
epoche consecutive da 4s, indipendentemente dalla ripetizione.

A differenza di simulatore.py (segnale sintetico, voto non sempre netto),
questo script garantisce un cambio di verdetto visibile ad ogni blocco —
utile per la demo al relatore.

Uso:
    python replay_real.py
    (poi, in un altro terminale: python prediction.py)
oppure tramite main_replay.py per la pipeline completa in un comando.
"""

import os
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

FS       = 256
N_CH     = 4
CH_NAMES = ["TP9", "AF7", "AF8", "TP10"]

BLOCK_DURATION = 10.0   # secondi per ciascun blocco neutral/concentrating

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPOCH_NEUTRAL_PATH      = os.path.join(BASE_DIR, "epoch_neutral.npy")
EPOCH_CONCENTRATING_PATH = os.path.join(BASE_DIR, "epoch_concentrating.npy")


def create_muse_outlet(name="MuseSim_1", source_id="muse_replay_001"):
    info = StreamInfo(name=name, type="EEG", channel_count=N_CH,
                       nominal_srate=FS, channel_format="float32",
                       source_id=source_id)
    desc = info.desc()
    desc.append_child_value("manufacturer", "RealEpochReplay")
    channels = desc.append_child("channels")
    for ch in CH_NAMES:
        c = channels.append_child("channel")
        c.append_child_value("label", ch)
        c.append_child_value("unit", "microvolts")
        c.append_child_value("type", "EEG")
    return StreamOutlet(info)


def run_replay(stop_event=None):
    """
    Pubblica in loop le due epoche reali, alternando blocchi da
    BLOCK_DURATION secondi. stop_event: threading.Event opzionale.
    """
    neutral = np.load(EPOCH_NEUTRAL_PATH)        # (1024, 4)
    conc    = np.load(EPOCH_CONCENTRATING_PATH)  # (1024, 4)

    outlet = create_muse_outlet()
    print(f"[REPLAY] Stream LSL avviato: MuseSim_1  ({FS} Hz, canali: {CH_NAMES})")
    print(f"[REPLAY] Blocchi da {BLOCK_DURATION:.0f}s: neutral / concentrating "
          f"(epoche reali da subjectb)\n")

    t0 = time.time()
    next_t = t0
    last_regime = None

    while stop_event is None or not stop_event.is_set():
        t = time.time() - t0
        regime = "neutral" if int(t // BLOCK_DURATION) % 2 == 0 else "concentrating"

        if regime != last_regime:
            print(f"[REPLAY] [t={t:6.1f}s]  regime → {regime}")
            last_regime = regime

        epoch = neutral if regime == "neutral" else conc

        # Indice del campione dentro l'epoca, ciclico (l'epoca si ripete)
        sample_idx = int((t * FS)) % len(epoch)
        sample = epoch[sample_idx].astype(np.float32)

        ts = local_clock()
        outlet.push_sample(sample.tolist(), ts)

        next_t += 1.0 / FS
        sleep_time = next_t - time.time()
        if sleep_time > 0:
            time.sleep(sleep_time)


def main():
    run_replay()


if __name__ == "__main__":
    main()
