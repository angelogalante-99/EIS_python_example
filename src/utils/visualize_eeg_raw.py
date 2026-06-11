"""
EEG Signal Visualizer — Dati Raw
---------------------------------
Carica eeg_dataset_raw.csv e visualizza per soggetto/stato:
  1. Segnale nel tempo (RawArray MNE, scorrevole con ← →)
  2. PSD media per canale (Welch, bande theta/alpha/beta evidenziate)

Uso:
    python visualize_eeg_raw.py
"""

import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import neurokit2 as nk

# ── Configurazione ───────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH  = os.path.join(BASE_DIR, "data", "eeg_dataset_raw.csv")
SFREQ     = 256.0
LABEL_MAP = {0: "neutral", 1: "concentrating"}

EEG_CHANNELS  = ["TP9", "AF7", "AF8", "TP10"]
PLOT_SCALINGS = {"eeg": 150e-6}
WINDOW_SEC    = 1   # finestra Welch (1 s → risoluzione 1 Hz)

CH_COLORS = {"TP9": "#185FA5", "AF7": "#D85A30", "AF8": "#1D9E75", "TP10": "#854F0B"}
BANDS     = [(4, 8, "blue", 0.07, "θ 4-8"), (8, 13, "green", 0.15, "α 8-13"),
             (13, 30, "red", 0.07, "β 13-30")]
# ─────────────────────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    print(f"\nCaricamento {path} ...")
    df = pd.read_csv(path)
    df["subject"] = df["subject"].str.replace(r".*/", "", regex=True)
    df["state"]   = df["label"].map(LABEL_MAP)
    return df


def build_raw(df_seg: pd.DataFrame) -> mne.io.RawArray:
    """Converte un segmento DataFrame in RawArray MNE (µV → V)."""
    data = df_seg[EEG_CHANNELS].values.T * 1e-6
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SFREQ, ch_types=["eeg"]*4)
    raw  = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"),
                    on_missing="ignore", verbose=False)
    return raw


def plot_psd(df_seg: pd.DataFrame, title: str):
    """
    Calcola e plotta la PSD media (Welch) per canale.
    - Asse X: frequenza (Hz) — NON è una durata temporale
    - Asse Y: potenza spettrale in dB/Hz
    - Bande theta/alpha/beta evidenziate come zone colorate
    - Una curva per canale: se è piatta in beta/gamma → EMG/artefatto
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for ch in EEG_CHANNELS:
        sig    = df_seg[ch].values
        psd_df = nk.signal_psd(sig, sampling_rate=SFREQ, method="welch",
                               show=False, window=WINDOW_SEC)
        psd_df = psd_df[psd_df["Frequency"] <= 40]
        psd_db = 10 * np.log10(np.maximum(psd_df["Power"].values, 1e-20))
        ax.plot(psd_df["Frequency"], psd_db, label=ch,
                color=CH_COLORS[ch], linewidth=1.6, alpha=0.85)

    for lo, hi, col, alpha, name in BANDS:
        ax.axvspan(lo, hi, color=col, alpha=alpha, label=name)

    ax.set_xlim(1, 40)
    ax.set_xlabel("Frequenza (Hz)  ←  non è una durata!")
    ax.set_ylabel("Potenza Spettrale (dB/Hz)")
    ax.set_title(f"PSD media — {title}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show(block=False)


def plot_sessions(df: pd.DataFrame, subject: str, label: int):
    state    = LABEL_MAP[label]
    sessions = sorted(df[(df["subject"] == subject) & (df["label"] == label)]
                      ["session"].unique())

    if not sessions:
        print(f"  Nessun dato per {subject} - {state}")
        return

    print(f"\n{'='*55}")
    print(f"  Soggetto : {subject}   Stato : {state}")
    print(f"  Sessioni : {sessions}")
    print(f"{'='*55}")

    for ses in sessions:
        seg = df[(df["subject"] == subject) & (df["label"] == label) &
                 (df["session"] == ses)].reset_index(drop=True)

        n_samples  = len(seg)
        duration_s = n_samples / SFREQ
        print(f"\n  Sessione {ses}: {n_samples:,} campioni "
              f"({duration_s:.1f} s ≈ {duration_s/60:.1f} min)")
        if duration_s < 30:
            print(f"  ⚠  Sessione molto corta ({duration_s:.1f} s)")

        title = f"{subject} | {state} | sessione {ses} ({duration_s:.0f} s)"

        # 1. Plot temporale — scorri con ← →
        raw = build_raw(seg)
        raw.plot(title=title, scalings=PLOT_SCALINGS,
                 n_channels=len(EEG_CHANNELS), duration=10.0, start=0.0,
                 block=False, show=True, bgcolor="white",
                 color={"eeg": "steelblue"})

        # 2. PSD media sulla stessa sessione
        plot_psd(seg, title)

    plt.show(block=True)


def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  RIEPILOGO DATASET RAW")
    print("="*60)
    ts_col = "timestamps" if "timestamps" in df.columns else df.columns[0]
    summary = (df.groupby(["subject", "state", "session"])[ts_col]
               .count().reset_index().rename(columns={ts_col: "campioni"}))
    summary["durata (s)"]   = (summary["campioni"] / SFREQ).round(1)
    summary["durata (min)"] = (summary["durata (s)"] / 60).round(2)
    print(summary.to_string(index=False))
    print(f"\nTotale campioni: {len(df):,}")
    short = summary[summary["durata (s)"] < 30]
    if not short.empty:
        print("\n⚠  Sessioni molto corte (< 30 s):")
        print(short[["subject", "state", "session", "campioni", "durata (s)"]].to_string(index=False))
    print("="*60)


def interactive_menu(df: pd.DataFrame):
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


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    try:
        df = load_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Errore: '{CSV_PATH}' non trovato.")
        raise
    print_summary(df)
    interactive_menu(df)