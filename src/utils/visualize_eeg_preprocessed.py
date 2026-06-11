"""
EEG Signal Visualizer — Dati Preprocessati
--------------------------------------------
Carica eeg_dataset_preprocessed.csv e visualizza per soggetto/stato:
  1. Epoche nel tempo (EpochsArray MNE, 5 epoche per volta)
  2. PSD media per canale (Welch, media su TUTTE le epoche del soggetto/stato)

Uso:
    python visualize_eeg_preprocessed.py
"""

import os
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import neurokit2 as nk

# ── Configurazione ───────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH  = os.path.join(BASE_DIR, "data", "eeg_dataset_preprocessed.csv")
SFREQ     = 256.0
LABEL_MAP = {0: "neutral", 1: "concentrating"}

EEG_CHANNELS  = ["TP9", "AF7", "AF8", "TP10"]
PLOT_SCALINGS = {"eeg": 150e-6}
WINDOW_SEC    = 1   # finestra Welch (1 s → risoluzione 1 Hz)

CH_COLORS = {"TP9": "#185FA5", "AF7": "#D85A30", "AF8": "#1D9E75", "TP10": "#854F0B"}
BANDS     = [(4, 8, "blue", 0.07, "θ 4-8"), (8, 13, "green", 0.15, "α 8-13"),
             (13, 30, "red", 0.07, "β 13-30")]
# ─────────────────────────────────────────────────────────────────────────────


def load_data(path: str) -> pd.DataFrame:
    print(f"\nCaricamento {path} ...")
    df = pd.read_csv(path)
    df["state"] = df["label"].map(LABEL_MAP)
    return df


def plot_mean_psd(df_subj_state: pd.DataFrame, title: str):
    """
    Calcola la PSD Welch su ogni epoca e plotta la MEDIA tra tutte le epoche.
    Media invece della singola epoca: più stabile, meno influenzata da artefatti
    impulsivi isolati.

    Come leggere il grafico:
    - Asse X: frequenza (Hz) — non è una durata temporale
    - Asse Y: potenza in dB/Hz — più è alta, più energia c'è a quella frequenza
    - Curva sana: decade da sinistra a destra (più potenza a basse frequenze)
    - Zona alpha (8-13 Hz, verde): piccola gobba = corteccia frontale a riposo
    - Zona beta (13-30 Hz, rossa): se la curva si appiattisce qui → EMG/artefatto
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    for ch in EEG_CHANNELS:
        psd_list = []
        for _, ep in df_subj_state.groupby(["session", "epoch_id"]):
            sig    = ep[ch].values
            psd_df = nk.signal_psd(sig, sampling_rate=SFREQ, method="welch",
                                   show=False, window=WINDOW_SEC)
            psd_df = psd_df[psd_df["Frequency"] <= 40]
            psd_list.append(psd_df.set_index("Frequency")["Power"])

        if not psd_list:
            continue

        import pandas as _pd
        mean_power = _pd.concat(psd_list, axis=1).ffill().mean(axis=1)
        psd_db     = 10 * np.log10(np.maximum(mean_power.values, 1e-20))
        n          = len(psd_list)
        ax.plot(mean_power.index, psd_db, label=f"{ch} (n={n})",
                color=CH_COLORS[ch], linewidth=1.6, alpha=0.85)

    for lo, hi, col, alpha, name in BANDS:
        ax.axvspan(lo, hi, color=col, alpha=alpha, label=name)

    ax.set_xlim(1, 40)
    ax.set_xlabel("Frequenza (Hz) ")
    ax.set_ylabel("Potenza Spettrale (dB/Hz)")
    ax.set_title(f"PSD media su tutte le epoche — {title}")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plt.show(block=False)


def plot_session_epochs(df: pd.DataFrame, subject: str, label: int):
    state      = LABEL_MAP[label]
    df_filt    = df[(df["subject"] == subject) & (df["label"] == label)]
    sessions   = sorted(df_filt["session"].unique())

    if not sessions:
        print(f"  Nessuna epoca disponibile per {subject} - {state}")
        return

    print(f"\n{'='*55}")
    print(f"  Soggetto : {subject}   Stato : {state}")
    print(f"  Sessioni : {sessions}")
    print(f"{'='*55}")

    for ses in sessions:
        df_ses = df_filt[df_filt["session"] == ses].reset_index(drop=True)

        epochs_data = []
        for _, group in df_ses.groupby("epoch_id", sort=False):
            epochs_data.append(group[EEG_CHANNELS].values.T * 1e-6)

        epochs_array = np.array(epochs_data)
        n_epochs     = epochs_array.shape[0]
        print(f"  Sessione {ses}: {n_epochs} epoche ({n_epochs * 4.0:.1f} s totali)")

        info   = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SFREQ, ch_types=["eeg"]*4)
        epochs = mne.EpochsArray(epochs_array, info, verbose=False)
        epochs.set_montage(mne.channels.make_standard_montage("standard_1020"),
                           on_missing="ignore", verbose=False)

        title = f"EPOCHE: {subject} | {state} | sessione {ses} ({n_epochs} epoche)"
        epochs.plot(n_epochs=5, n_channels=len(EEG_CHANNELS),
                    scalings=PLOT_SCALINGS, title=title, block=False, show=True)

    # PSD media su TUTTE le sessioni/epoche del soggetto+stato
    plot_mean_psd(df_filt, f"{subject} | {state}")
    plt.show(block=True)


def print_summary(df: pd.DataFrame):
    print("\n" + "="*60)
    print("  RIEPILOGO EPOCHE PULITE NEL DATASET")
    print("="*60)
    summary = (df.groupby(["subject", "state", "session"])["epoch_id"]
               .nunique().reset_index()
               .rename(columns={"epoch_id": "epoche_totali"}))
    summary["tempo_pulito (s)"] = summary["epoche_totali"] * 4.0
    print(summary.to_string(index=False))
    print(f"\nTotale epoche: {df.groupby(['subject','session','label','epoch_id']).ngroups}")
    print("="*60)


def interactive_menu(df: pd.DataFrame):
    subjects = sorted(df["subject"].unique())
    labels   = sorted(df["label"].unique())

    while True:
        print("\nSoggetti disponibili (dati preprocessati):")
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

        plot_session_epochs(df, subject, label)


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    try:
        df = load_data(CSV_PATH)
    except FileNotFoundError:
        print(f"Errore: '{CSV_PATH}' non trovato.")
        raise
    print_summary(df)
    interactive_menu(df)