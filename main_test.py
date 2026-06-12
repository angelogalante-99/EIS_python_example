"""
Main (Dataset esterno) — Pipeline completa offline
-----------------------------------------------------------------
Stesso schema di main.py (preprocessing -> features_extraction ->
prediction), ma applicato OFFLINE su un volontario a scelta del
dataset esterno Muse2-eeg (yueeyue/Muse2-eeg), invece che su uno
stream LSL.

Passi:
  1. Scegli un volontario (VOLUNTEER) e le attività da includere
  2. I CSV del volontario vengono letti e portati nella STESSA
     struttura tabellare di eeg_dataset_raw.csv:
       subject, session, timestamps, TP9, AF7, AF8, TP10, label
     (stessa logica di convert_external_dataset.py, ma in memoria,
     per un solo volontario)
  3. Per ogni sessione, il segnale viene tagliato in epoche da 4s
     (1024 campioni, NESSUN overlap — coerente con lo streaming reale)
  4. Per ogni epoca:
       preprocessing.preprocess_epoch      -> filtri + trial rejection
       features_extraction.extract_features -> 6 feature
       prediction.predict_epoch             -> voto di maggioranza
  5. Riepilogo finale: confronto verdetto vs label "proxy"
     (rest=neutral, read/game=concentrating)

NOTA: rest/read/game come proxy di neutral/concentrating sono un
protocollo diverso dal nostro (vedi convert_external_dataset.py).
Questo è un test di generalizzazione fuori-dominio, non una LOSO.

Setup:
  Scarica ed estrai Muse2-dataset.zip (README di
  https://github.com/yueeyue/Muse2-eeg) in EXTERNAL_DIR
  (struttura attesa: EXTERNAL_DIR/volunteer_X/<activity>_<ts>.csv)

Uso:
    python main_external.py
"""

import os
import sys
import glob
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import mne
mne.set_log_level("WARNING")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src", "ml"))
sys.path.insert(0, os.path.join(ROOT, "src", "Preprocessing"))
sys.path.insert(0, os.path.join(ROOT, "src", "features"))

import prediction
from preprocessing import preprocess_epoch
from features_extraction import extract_features, FEATURE_NAMES

# ── CONFIGURA QUI ────────────────────────────────────────────────────────────
EXTERNAL_DIR = os.path.join(ROOT, "Muse2 dataset")

# Quale volontario testare (cartella external_data/volunteer_X)
VOLUNTEER = "volunteer 8"

EPOCH_LEN = 1024   # 4s @ 256Hz — stesso formato del nostro progetto

COL_MAP  = {"TP9": "RAW_TP9", "AF7": "RAW_AF7", "AF8": "RAW_AF8", "TP10": "RAW_TP10"}
CH_ORDER = ["TP9", "AF7", "AF8", "TP10"]

# rest -> neutral (0), read/game -> concentrating (1); eat/toy/tv esclusi
ACTIVITY_LABEL = {"rest": 0, "read": 1, "game": 1}
# ─────────────────────────────────────────────────────────────────────────────


def load_volunteer_as_raw_table(volunteer_dir):
    """
    Legge tutti i CSV di un volontario e li porta nella STESSA struttura
    tabellare di eeg_dataset_raw.csv:
        subject, session, timestamps, TP9, AF7, AF8, TP10, label

    Ritorna un DataFrame unico (tutte le sessioni concatenate).
    """
    subject = "ext_" + os.path.basename(volunteer_dir)
    csv_files = sorted(glob.glob(os.path.join(volunteer_dir, "*.csv")))

    rows = []
    session_counter = {}

    for csv_path in csv_files:
        fname = os.path.basename(csv_path).lower()
        activity = fname.split("_")[0]
        if activity not in ACTIVITY_LABEL:
            continue

        df = pd.read_csv(csv_path, low_memory=False)
        raw_cols = list(COL_MAP.values())
        if any(c not in df.columns for c in raw_cols):
            continue

        for c in raw_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=raw_cols).reset_index(drop=True)

        if len(df) < EPOCH_LEN:
            print(f"  [{fname}] scartato: {len(df)} campioni (< {EPOCH_LEN})")
            continue

        out = pd.DataFrame({ch: df[COL_MAP[ch]].values for ch in CH_ORDER})
        out["timestamps"] = np.arange(len(out)) / 256.0

        session_counter[activity] = session_counter.get(activity, 0) + 1
        out["subject"] = subject
        out["session"] = f"{activity}{session_counter[activity]}"
        out["label"]   = ACTIVITY_LABEL[activity]

        rows.append(out)
        print(f"  [{fname}] activity={activity} label={ACTIVITY_LABEL[activity]} "
              f"-> {len(out)} campioni ({len(out)/256:.1f}s)")

    if not rows:
        return pd.DataFrame()

    final_df = pd.concat(rows, ignore_index=True)
    return final_df[["subject", "session", "timestamps"] + CH_ORDER + ["label"]]


def epoch_generator_from_table(df):
    """
    Genera epoche consecutive (1024 x 4, no overlap) da un DataFrame
    raw_table, una sessione alla volta. Yield: (subject, session, label,
    epoch_idx, epoch array).
    """
    for (subject, session, label), group in df.groupby(["subject", "session", "label"]):
        signal = group[CH_ORDER].values
        n_epochs = len(signal) // EPOCH_LEN
        for i in range(n_epochs):
            epoch = signal[i*EPOCH_LEN:(i+1)*EPOCH_LEN]
            yield subject, session, label, i, epoch


def main():
    print("=" * 60)
    print("  PIPELINE COMPLETA (DATASET ESTERNO, offline)")
    print("  preprocessing -> features_extraction -> prediction")
    print("=" * 60)

    volunteer_dir = os.path.join(EXTERNAL_DIR, VOLUNTEER)
    if not os.path.isdir(volunteer_dir):
        print(f"\nCartella non trovata: {volunteer_dir}")
        print("Scarica/estrai Muse2-dataset.zip in external_data/ "
              "(vedi docstring) e/o cambia VOLUNTEER.")
        return

    print(f"\nVolontario: {VOLUNTEER}")
    print(f"Conversione CSV -> struttura raw_table ...")
    raw_table = load_volunteer_as_raw_table(volunteer_dir)

    if raw_table.empty:
        print("Nessun dato valido per questo volontario.")
        return

    print(f"\nRaw table: {len(raw_table):,} righe, "
          f"sessioni: {sorted(raw_table['session'].unique())}")

    print("\nCaricamento modelli (model_final.pkl)...")
    models = prediction.load_models()

    print(f"\nElaborazione epoche da {EPOCH_LEN} campioni (4s @ 256Hz)...\n")

    results = []
    for subject, session, label, epoch_idx, raw_epoch in epoch_generator_from_table(raw_table):
        # 1) PREPROCESSING
        filtered_epoch, ok = preprocess_epoch(raw_epoch)
        if not ok:
            print(f"[{session} #{epoch_idx}]  SCARTATA (artefatto > soglia)")
            continue

        # 2) FEATURES_EXTRACTION
        epoch_signals = {
            "AF7": filtered_epoch[:, CH_ORDER.index("AF7")],
            "AF8": filtered_epoch[:, CH_ORDER.index("AF8")],
        }
        feat_dict = extract_features(epoch_signals)
        features  = [feat_dict[name] for name in FEATURE_NAMES]

        # 3) PREDICTION
        verdict, details = prediction.predict_epoch(models, features)

        vote_str = "  ".join(
            f"{name.upper()}={'C' if v==1 else 'N'}"
            + (f"({details['probs'][name]*100:.0f}%)" if name in details['probs'] else "")
            for name, v in details["votes"].items()
        )

        label_str   = prediction.LABEL_MAP[label]
        verdict_str = prediction.LABEL_MAP[verdict]
        match = "OK " if verdict == label else "ERR"

        print(f"[{session:<8} #{epoch_idx:3d}]  proxy={label_str:<13}  "
              f"{vote_str}  ->  {verdict_str.upper():<13}  [{match}]")

        results.append({"session": session, "label": label, "verdict": verdict})

    # ── Riepilogo ────────────────────────────────────────────────────────────
    if not results:
        print("\nNessuna epoca valida elaborata.")
        return

    res = pd.DataFrame(results)
    acc = (res["label"] == res["verdict"]).mean()

    print(f"\n{'='*60}")
    print(f"  RIEPILOGO — {VOLUNTEER}")
    print(f"{'='*60}")
    print(f"  Epoche elaborate : {len(res)}")
    print(f"  Accuracy (proxy) : {acc*100:.1f}%")
    print(f"\n  Per sessione:")
    for session, g in res.groupby("session"):
        acc_s = (g["label"] == g["verdict"]).mean()
        print(f"    {session:<10} n={len(g):3d}  accuracy={acc_s*100:5.1f}%")

    print(f"\n  NOTA: 'accuracy (proxy)' confronta il verdetto con rest=neutral,")
    print(f"  read/game=concentrating — un protocollo diverso dal training.")
    print(f"  Da interpretare come test di generalizzazione fuori-dominio,")
    print(f"  non come una nuova LOSO.")


if __name__ == "__main__":
    main()