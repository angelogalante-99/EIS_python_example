"""
Main — Orchestratore della pipeline completa
-------------------------------------------------------------------
Per ogni epoca proveniente dallo stream LSL, applica in sequenza
TUTTI i moduli del progetto, ciascuno importato dalla sua cartella:

  streaming           (src/ml)        -> epoca grezza dallo stream (1024x4)
  preprocessing       (src/Preprocessing) -> notch + band-pass + trial rejection
  features_extraction (src/features)  -> 6 feature (Abs/Rel Theta/Alpha, FAA)
  prediction          (src/ml)        -> voto di maggioranza (SVM/LDA/LightGBM)

In parallelo, in un thread separato, replay_real.py pubblica su LSL
epoche REALI (neutral/concentrating alternate ogni 10s) per la demo.

Uso:
    python main.py
"""

import os
import sys
import threading
import time
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import mne
mne.set_log_level("WARNING")

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Import dai moduli del progetto, ciascuno dalla sua cartella ─────────────
sys.path.insert(0, os.path.join(ROOT, "src", "ml"))
sys.path.insert(0, os.path.join(ROOT, "src", "Preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "src", "features"))

import streaming                                   # src/ml/streaming.py
import prediction                                  # src/ml/prediction.py
from preprocessing import preprocess_epoch         # src/Preprocessing/preprocessing.py
from features_extraction import extract_features, FEATURE_NAMES  # src/features/features_extraction.py

import replay_real   # generatore epoche reali per la demo (in root)


def main():
    print("=" * 60)
    print("  PIPELINE COMPLETA")
    print("  streaming -> preprocessing -> features_extraction -> prediction")
    print("=" * 60)

    stop_event = threading.Event()

    # ── Avvia il replay di epoche reali in un thread separato ────────────────
    replay_thread = threading.Thread(
        target=replay_real.run_replay,
        args=(stop_event,),
        daemon=True,
    )
    replay_thread.start()
    time.sleep(1.0)  # tempo per pubblicare lo stream LSL

    # ── Setup: connessione + modelli ──────────────────────────────────────────
    inlet  = streaming.connect_lsl()

    print("\nCaricamento modelli (model_final.pkl)...")
    models = prediction.load_models()

    print(f"\nAccumulo epoche da {streaming.EPOCH_SEC:.0f}s "
          f"({streaming.EPOCH_LEN} campioni @ {streaming.SFREQ:.0f} Hz)...\n")

    # ── Loop principale ────────────────────────────────────────────────────────
    try:
        for epoch_n, raw_epoch in enumerate(streaming.epoch_generator(inlet, stop_event), 1):

            # 1) STREAMING -> epoca grezza (1024 x 4): già ricevuta come raw_epoch

            # 2) PREPROCESSING -> filtri notch+band-pass, trial rejection
            filtered_epoch, ok = preprocess_epoch(raw_epoch)
            if not ok:
                print(f"epoca #{epoch_n:4d}  SCARTATA (artefatto > soglia)")
                continue

            # 3) FEATURES_EXTRACTION -> 6 feature (Abs/Rel Theta/Alpha AF7/AF8, FAA)
            epoch_signals = {
                "AF7": filtered_epoch[:, streaming.CH_NAMES.index("AF7")],
                "AF8": filtered_epoch[:, streaming.CH_NAMES.index("AF8")],
            }
            feat_dict = extract_features(epoch_signals)
            features  = [feat_dict[name] for name in FEATURE_NAMES]

            # 4) PREDICTION -> voto di maggioranza (SVM/LDA/LightGBM)
            verdict, details = prediction.predict_epoch(models, features)

            # ── Output ────────────────────────────────────────────────────────
            vote_str = "  ".join(
                f"{name.upper()}={'C' if v==1 else 'N'}"
                + (f"({details['probs'][name]*100:.0f}%)" if name in details['probs'] else "")
                for name, v in details["votes"].items()
            )
            ts_str = time.strftime("%H:%M:%S")
            print(f"[{ts_str}] epoca #{epoch_n:4d}  "
                  f"{vote_str}  →  "
                  f"VERDETTO: {prediction.LABEL_MAP[verdict].upper()} "
                  f"({details['n_concentrating']}/{details['n_models']} voti)")

    except KeyboardInterrupt:
        print("\n\nInterruzione richiesta — chiusura in corso...")
    finally:
        stop_event.set()
        replay_thread.join(timeout=2.0)
        print("Pipeline arrestata.")


if __name__ == "__main__":
    main()