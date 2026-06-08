"""
Visualizzatore Epoche EEG Interattivo (Dati Preprocessati)
---------------------------------------------------------
Carica eeg_dataset_preprocessed.csv e permette di ispezionare
le epoche pulite sessione per sessione tramite un menu interattivo.

Uso:
    python visualize_eeg_preprocessed.py
"""

import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt

# ── Configurazione ──────────────────────────────────────────
CSV_PATH     = "eeg_dataset_preprocessed.csv"
SFREQ        = 256.0                          # Hz
LABEL_MAP    = {0: "neutral", 1: "concentrating"}

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
PLOT_SCALINGS = {"eeg": 150e-6}
# ────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """Carica il dataset preprocessed."""
    print(f"\nCaricamento {path}...")
    df = pd.read_csv(path)
    df["state"] = df["label"].map(LABEL_MAP)
    return df

def print_summary(df: pd.DataFrame):
    """Mostra un riepilogo di quante epoche pulite ci sono per ogni sessione."""
    print("\n" + "="*60)
    print("  RIEPILOGO EPOCHE PULITE NEL DATASET")
    print("="*60)
    
    # Conta quante epoche uniche ci sono per ogni combinazione
    summary = (
        df.groupby(["subject", "state", "session"])["epoch_id"]
        .nunique()
        .reset_index()
        .rename(columns={"epoch_id": "epoche_totali"})
    )
    
    # Calcola i secondi totali di dati puliti (ogni epoca dura 4 secondi)
    summary["tempo_pulito (s)"] = summary["epoche_totali"] * 4.0
    
    print(summary.to_string(index=False))
    print(f"\nNumero totale di epoche nel file: {df.groupby(['subject', 'session', 'label', 'epoch_id']).ngroups}")
    print("="*60)

def plot_session_epochs(df: pd.DataFrame, subject: str, label: int):
    """Isola e plotta le epoche di un soggetto e stato specifico, divise per sessione."""
    state = LABEL_MAP[label]
    
    # Trova le sessioni disponibili per questa combinazione
    df_filtered = df[(df["subject"] == subject) & (df["label"] == label)]
    sessions = sorted(df_filtered["session"].unique())
    
    if not sessions:
        print(f"  Nessuna epoca disponibile per {subject} - {state}")
        return

    print(f"\n{'='*55}")
    print(f"  Soggetto : {subject}")
    print(f"  Stato    : {state}  (label={label})")
    print(f"  Sessioni : {sessions}")
    print(f"{'='*55}")

    for ses in sessions:
        # Isola i campioni di questa specifica sessione
        df_ses = df_filtered[df_filtered["session"] == ses].reset_index(drop=True)
        
        # Estrai le epoche mantendo la struttura 3D richiesta da MNE: (n_epoche, n_canali, n_campioni)
        epochs_data = []
        
        # Raggruppiamo per ID epoca per ricostruire i blocchi da 4 secondi
        for ep_id, group in df_ses.groupby("epoch_id", sort=False):
            # Convertiamo in Volt (MNE standard)
            data_matrix = group[EEG_CHANNELS].values.T * 1e-6 
            epochs_data.append(data_matrix)
            
        epochs_array = np.array(epochs_data)
        n_epochs = epochs_array.shape[0]
        
        print(f"  Sessione {ses}: Trovate {n_epochs} epoche pulite ({n_epochs * 4.0:.1f} secondi totali di segnale)")
        
        # Creazione dell'oggetto MNE Epochs
        ch_types = ["eeg"] * len(EEG_CHANNELS)
        info = mne.create_info(ch_names=EEG_CHANNELS, sfreq=SFREQ, ch_types=ch_types)
        epochs = mne.EpochsArray(epochs_array, info, verbose=False)
        
        # Applica il montaggio dei sensori
        montage = mne.channels.make_standard_montage("standard_1020")
        epochs.set_montage(montage, on_missing="ignore", verbose=False)
        
        # Genera il titolo per la finestra grafica
        title = f"EPOCHE PULITE: {subject} | {state} | Sessione {ses} ({n_epochs} epoche)"
        
        # Plot (Versione universale e sicura senza conflitti di colore)
        epochs.plot(
            n_epochs=5,                           
            n_channels=len(EEG_CHANNELS),           
            scalings=PLOT_SCALINGS,                        
            title=title,
            block=False,                       
            show=True
        )
        
    plt.show(block=True)

def interactive_menu(df: pd.DataFrame):
    """Menu interattivo per selezionare i dati da ispezionare."""
    subjects = sorted(df["subject"].unique())
    labels   = sorted(df["label"].unique())

    while True:
        print("\nSoggetti disponibili nel dataset preprocessato:")
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

def main():
    try:
        df = load_data(CSV_PATH)
    except FileNotFoundError:
        print(f"Errore: Il file '{CSV_PATH}' non esiste. Esegui prima il preprocessing!")
        return
        
    print_summary(df)
    interactive_menu(df)

if __name__ == "__main__":
    mne.set_log_level("WARNING")
    main()