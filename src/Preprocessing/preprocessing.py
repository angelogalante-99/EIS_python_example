"""
EEG Signal Pre-processing Pipeline
----------------------------------
Legge i dati raw, applica filtri (Notch + Band-pass),
esegue l'epoching con OVERLAPPING (finestre di 4s, step 1s)
e rimuove gli artefatti tramite TRIAL REJECTION (soglia alta).
"""

import pandas as pd
import numpy as np
import mne
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV = os.path.join(BASE_DIR, "data", "eeg_dataset_raw.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "eeg_dataset_preprocessed.csv")

SFREQ = 256.0  # Hz
EPOCH_LEN_S = 4.0  # Lunghezza di ogni epoca in secondi
STEP_LEN_S = 1.0  # Overlapping: la finestra avanza di 1 secondo

SAMPLES_EPOCH = int(SFREQ * EPOCH_LEN_S)
SAMPLES_STEP = int(SFREQ * STEP_LEN_S)

# Filtri
NOTCH_FREQ = 50.0  # Rumore di rete europeo (Hz)
BANDPASS_LOW = 1.0  # Taglio basso (Hz) - rimuove deriva lenta
BANDPASS_HIGH = 40.0  # Taglio alto (Hz) - rimuove rumore muscolare ad alta freq.

# Trial Rejection
ARTEFACT_TH = 500e-6  # 300 µV

LABEL_MAP = {0: "neutral", 1: "concentrating"}
EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


# ────────────────────────────────────────────────────────────


def preprocess_epoch(epoch):
    """
    Applica notch (50Hz) + band-pass (1-40Hz) a una singola epoca
    e verifica la soglia di rigetto artefatti.

    epoch: array numpy (SAMPLES_EPOCH, 4) in µV, colonne EEG_CHANNELS

    Ritorna: (epoch_filtrata, ok)
      epoch_filtrata: stessa shape, dopo i filtri (in µV)
      ok: True se l'epoca passa la trial rejection, False se va scartata

    Stessa logica di process_session() qui sotto, applicata a 4s
    invece che a un'intera sessione con overlapping — è la versione
    "real-time" della stessa pipeline.

    NOTA: usa filtri IIR (non FIR come process_session) perché i filtri
    FIR di default di MNE richiedono più campioni di quanti ne ha
    un'epoca da 4s (1024) — su una sessione intera (batch) non è un
    problema, su una singola epoca lo è.
    """
    data = epoch.T * 1e-6  # µV -> V, shape (4, n_samples)
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SFREQ, ch_types=["eeg"] * 4)
    raw = mne.io.RawArray(data, info, verbose=False)

    raw.notch_filter(freqs=NOTCH_FREQ, picks="eeg", method="iir", verbose=False)
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, picks="eeg",
               method="iir", verbose=False)

    filtered = raw.get_data().T * 1e6  # V -> µV, shape (n_samples, 4)

    ok = np.max(np.abs(filtered)) <= (ARTEFACT_TH * 1e6)
    return filtered, ok


def clean_short_sessions(df: pd.DataFrame, min_duration_s: int = 8) -> pd.DataFrame:
    """Rimuove le sessioni che durano meno della soglia specificata."""
    print(f"\n--- Fase 1: Pulizia Dataset ---")
    min_samples = min_duration_s * SFREQ

    # Calcola la dimensione di ogni sessione
    session_counts = df.groupby(["subject", "session", "label"]).size().reset_index(name="counts")

    # Identifica le sessioni troppo corte
    short_sessions = session_counts[session_counts["counts"] < min_samples]

    if not short_sessions.empty:
        print(f"Rimozione di {len(short_sessions)} sessioni troppo corte (< {min_duration_s}s):")
        for _, row in short_sessions.iterrows():
            print(
                f"  Scartata: {row['subject']} | sessione {row['session']} | stato: {LABEL_MAP[row['label']]} ({row['counts'] / SFREQ:.1f}s)")

        # Filtra il dataframe originale mantenendo solo le sessioni "lunghe"
        valid_sessions = session_counts[session_counts["counts"] >= min_samples]
        df_clean = pd.merge(df, valid_sessions[["subject", "session", "label"]], on=["subject", "session", "label"])
        return df_clean
    else:
        print("Nessuna sessione anomala rilevata.")
        return df


def process_session(df_session: pd.DataFrame) -> pd.DataFrame:
    """Applica filtri, epoching con overlapping e rigetto artefatti a una singola sessione."""
    # 1. Costruzione MNE RawArray
    data = df_session[EEG_CHANNELS].values.T * 1e-6  # Da µV a V
    info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SFREQ, ch_types=["eeg"] * 4)
    raw = mne.io.RawArray(data, info, verbose=False)

    # 2. Filtraggio Spaziale e Frequenziale
    # Notch filter (50Hz)
    raw.notch_filter(freqs=NOTCH_FREQ, picks="eeg", verbose=False)
    # Band-pass filter (1-40Hz)
    raw.filter(l_freq=BANDPASS_LOW, h_freq=BANDPASS_HIGH, picks="eeg", verbose=False)

    # Ritorno a Pandas per Epoching manuale
    filtered_data = raw.get_data().T * 1e6  # Riportiamo in µV

    n_samples = len(filtered_data)
    epochs = []
    epoch_counter = 0

    # 3. Epoching con Overlapping (Avanzamento a step di 1 secondo)
    for start_idx in range(0, n_samples - SAMPLES_EPOCH + 1, SAMPLES_STEP):
        end_idx = start_idx + SAMPLES_EPOCH

        epoch_data = filtered_data[start_idx:end_idx, :]

        # 4. Trial Rejection (Controllo Soglia a 250 µV)
        if np.max(np.abs(epoch_data)) > (ARTEFACT_TH * 1e6):
            continue  # Salta questa epoca, artefatto estremo rilevato!

        # Trasforma l'epoca pulita in DataFrame temporaneo
        df_epoch = pd.DataFrame(epoch_data, columns=EEG_CHANNELS)
        df_epoch["epoch_id"] = epoch_counter  # Identificativo sequenziale dell'epoca
        epochs.append(df_epoch)
        epoch_counter += 1

    if not epochs:
        return pd.DataFrame()  # Tutte le epoche sono state scartate

    df_final_session = pd.concat(epochs, ignore_index=True)
    return df_final_session


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Errore: Il file {INPUT_CSV} non esiste.")
        return

    df_raw = pd.read_csv(INPUT_CSV)

    # 1. Rimuovi sessioni corte (< 8s)
    df_clean = clean_short_sessions(df_raw, min_duration_s=8)

    print(f"\n--- Fase 2: Filtraggio, Overlapping e Trial Rejection ---")
    print(f"Finestra: {EPOCH_LEN_S}s | Avanzamento: {STEP_LEN_S}s")
    print(f"Soglia Rigetto: ±{ARTEFACT_TH * 1e6:.0f} µV")
    print("-" * 50)

    processed_sessions = []

    # Raggruppa per sessione logica
    groups = df_clean.groupby(["subject", "session", "label"])

    for (subject, session, label), df_group in groups:
        state_name = LABEL_MAP[label]

        # Calcolo matematico delle finestre teoriche possibili con overlapping
        teoric_windows = (len(df_group) - SAMPLES_EPOCH) // SAMPLES_STEP + 1

        df_processed = process_session(df_group)

        if df_processed.empty:
            print(f"[{subject} - {state_name} - Ses {session}] SCARTATA: 100% epoche inquinate.")
            continue

        # Calcolo Tasso di Sopravvivenza
        surviving_windows = df_processed["epoch_id"].nunique()

        print(f"[{subject} - {state_name} - Ses {session}] OK! Salvate {surviving_windows}/{teoric_windows} epoche")

        # Aggiungi metadati
        df_processed["subject"] = subject
        df_processed["session"] = session
        df_processed["label"] = label

        processed_sessions.append(df_processed)

    print("-" * 50)

    if not processed_sessions:
        print("Errore critico: Nessuna sessione è sopravvissuta al preprocessing.")
        return

    # Unisci tutto
    final_df = pd.concat(processed_sessions, ignore_index=True)

    # Riordina colonne (Metadati -> Dati -> Label)
    cols = ["subject", "session", "epoch_id"] + EEG_CHANNELS + ["label"]
    final_df = final_df[cols]

    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSalvataggio completato: {OUTPUT_CSV} ({len(final_df):,} righe totali).")


if __name__ == "__main__":
    mne.set_log_level("WARNING")
    main()