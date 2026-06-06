"""
EEG Signal Visualizer con MNE
-------------------------------
Carica eeg_dataset_final.csv e visualizza i segnali raw per
soggetto, label e sessione. Permette di ispezionare
visivamente artefatti, saturazioni e anomalie.

Uso:
    python visualize_eeg.py

Dipendenze:
    pip install mne pandas numpy matplotlib
"""

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

# ── Configurazione ──────────────────────────────────────────
CSV_PATH  = r"eeg_dataset_final.csv"   # cambia con il percorso del tuo file

SFREQ     = 256.0                      # frequenza di campionamento Muse (Hz)

LABEL_MAP = {0: "neutral", 1: "concentrating"}  # mappatura label → nome

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
AUX_CHANNELS = ["Right AUX"]
ALL_CHANNELS = EEG_CHANNELS + AUX_CHANNELS

# Scala di visualizzazione — abbassala se i segnali escono dallo schermo
PLOT_SCALINGS = {"eeg": 150e-6, "misc": 150e-6}
# ────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    print(f"\nCaricamento {path} ...")
    df = pd.read_csv(path)
    df["subject"] = df["subject"].str.replace(r".*/", "", regex=True)
    df["state"]   = df["label"].map(LABEL_MAP)
    return df


def build_raw(df_segment: pd.DataFrame) -> mne.io.RawArray:
    """Converte un segmento DataFrame in un oggetto MNE RawArray."""
    data = df_segment[ALL_CHANNELS].values.T * 1e-6  # µV → V

    ch_types = ["eeg"] * len(EEG_CHANNELS) + ["misc"] * len(AUX_CHANNELS)
    info = mne.create_info(
        ch_names=ALL_CHANNELS,
        sfreq=SFREQ,
        ch_types=ch_types,
    )
    raw = mne.io.RawArray(data, info, verbose=False)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage, on_missing="ignore", verbose=False)
    return raw


def plot_sessions(df: pd.DataFrame, subject: str, label: int):
    """
    Plotta sessione 1 e sessione 2 dello stesso soggetto/label
    in due finestre separate per confronto diretto.
    """
    state = LABEL_MAP[label]
    sessions = sorted(df[
        (df["subject"] == subject) & (df["label"] == label)
    ]["session"].unique())

    if not sessions:
        print(f"  Nessun dato per {subject} - {state}")
        return

    print(f"\n{'='*55}")
    print(f"  Soggetto : {subject}")
    print(f"  Stato    : {state}  (label={label})")
    print(f"  Sessioni : {sessions}")
    print(f"{'='*55}")

    for ses in sessions:
        seg = df[
            (df["subject"] == subject) &
            (df["label"]   == label)   &
            (df["session"] == ses)
        ].reset_index(drop=True)

        n_samples  = len(seg)
        duration_s = n_samples / SFREQ

        print(f"\n  Sessione {ses}: {n_samples:,} campioni  "
              f"({duration_s:.1f} s  ≈  {duration_s/60:.1f} min)")

        if duration_s < 30:
            print(f"  ⚠  ATTENZIONE: sessione molto corta "
                  f"({duration_s:.1f} s) — valuta se scartarla!")

        raw = build_raw(seg)

        title = (f"{subject}  |  {state}  |  sessione {ses}  "
                 f"({n_samples:,} campioni, {duration_s:.0f} s)")

        raw.plot(
            title=title,
            scalings=PLOT_SCALINGS,
            n_channels=len(ALL_CHANNELS),
            duration=10.0,      # secondi visibili per volta — usa ← → per scorrere
            start=0.0,
            block=False,
            show=True,
            bgcolor="white",
            color={"eeg": "steelblue", "misc": "darkorange"},
        )

    plt.show(block=True)


def print_summary(df: pd.DataFrame):
    """Riepilogo dataset con durate e segnalazione sessioni anomale."""
    print("\n" + "="*60)
    print("  RIEPILOGO DATASET")
    print("="*60)

    summary = (
        df.groupby(["subject", "state", "session"])["timestamps"]
        .count()
        .reset_index()
        .rename(columns={"timestamps": "campioni"})
    )
    summary["durata (s)"]   = (summary["campioni"] / SFREQ).round(1)
    summary["durata (min)"] = (summary["durata (s)"] / 60).round(2)
    print(summary.to_string(index=False))
    print(f"\nTotale campioni: {len(df):,}")

    short = summary[summary["durata (s)"] < 30]
    if not short.empty:
        print("\n⚠  Sessioni molto corte (< 30 s) — da valutare:")
        print(short[["subject","state","session","campioni","durata (s)"]].to_string(index=False))
    print("="*60)


def interactive_menu(df: pd.DataFrame):
    """Menu per scegliere soggetto e stato da visualizzare."""
    subjects = sorted(df["subject"].unique())
    labels   = sorted(df["label"].unique())

    while True:
        print("\nSoggetti disponibili:")
        for i, s in enumerate(subjects, 1):
            print(f"  {i}. {s}")
        print("  0. Esci")

        choice = input("\nScegli soggetto (numero): ").strip()
        if choice == "0":
            print("Uscita.")
            break
        try:
            subject = subjects[int(choice) - 1]
        except (ValueError, IndexError):
            print("  Scelta non valida, riprova.")
            continue

        print(f"\nStati disponibili per {subject}:")
        for i, lbl in enumerate(labels, 1):
            print(f"  {i}. {LABEL_MAP[lbl]}  (label={lbl})")
        print("  0. Torna indietro")

        choice2 = input("\nScegli stato (numero): ").strip()
        if choice2 == "0":
            continue
        try:
            label = labels[int(choice2) - 1]
        except (ValueError, IndexError):
            print("  Scelta non valida, riprova.")
            continue

        plot_sessions(df, subject, label)


# ── Entry point ─────────────────────────────────────────────
if __name__ == "__main__":
    mne.set_log_level("WARNING")

    df = load_csv(CSV_PATH)
    print_summary(df)
    interactive_menu(df)