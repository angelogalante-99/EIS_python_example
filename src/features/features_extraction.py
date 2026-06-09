"""
EEG Feature Extraction Pipeline (PSD & Ratios)
----------------------------------------------
Calcola la Power Spectral Density (PSD) tramite il metodo di Welch
ed estrae le features in banda (Theta, Alpha, Beta) usando NeuroKit2.
"""

import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import os

# ── Configurazione ──────────────────────────────────────────
INPUT_CSV  = "data/eeg_dataset_preprocessed.csv"  # soglia globale 300-500uV
OUTPUT_CSV = "data/final_features.csv"

# Impostiamo il percorso corretto per i grafici: cartella "features", sottocartella "sanity_checks"
SANITY_DIR = os.path.join("src", "features", "sanity_checks")

SFREQ    = 256.0
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# NeuroKit ragiona in secondi per la grandezza della finestra di Welch
WINDOW_SEC = 1   # finestre interne da 1s, PSD stabile con risoluzione 1Hz
# ────────────────────────────────────────────────────────────

def save_sanity_check(psd_store, subject, state, out_dir=SANITY_DIR):
    """Salva il grafico PSD su disco utilizzando i DataFrame generati da NeuroKit2."""
    # Ora out_dir punta a "features/sanity_checks". os.makedirs creerà entrambe le cartelle se mancano.
    os.makedirs(out_dir, exist_ok=True)
    subj_short = subject.split("/")[-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    for ch in CHANNELS:
        # nk.signal_psd restituisce un DataFrame. Estraiamo Frequenza e Potenza.
        psd_df = psd_store[ch]
        
        # Limitiamo il grafico a 40 Hz per leggibilità
        psd_plot = psd_df[psd_df["Frequency"] <= 40.0]
        
        # Trasformazione logaritmica (dB) per visualizzazione
        psd_db = 10 * np.log10(np.maximum(psd_plot["Power"], 1e-20))  
        ax.plot(psd_plot["Frequency"], psd_db, label=ch, alpha=0.85)

    ax.axvspan(8,  13, color="green", alpha=0.15, label="Alpha (8-13 Hz)")
    ax.axvspan(13, 30, color="red",   alpha=0.07, label="Beta  (13-30 Hz)")
    ax.axvspan(4,  8,  color="blue",  alpha=0.07, label="Theta (4-8 Hz)")

    ax.set_title(f"Sanity Check PSD (NeuroKit2) — {subj_short} | {state}")
    ax.set_xlabel("Frequenza (Hz)")
    ax.set_ylabel("Potenza Spettrale (dB/Hz)")
    ax.set_xlim(1, 40)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    # Uniamo la cartella di destinazione col nome del file
    out_path = os.path.join(out_dir, f"{subj_short}_{state}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Sanity check salvato: {out_path}")


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Errore: {INPUT_CSV} non trovato.")
        return

    print(f"Caricamento {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  {len(df):,} righe  |  epoche: {df.groupby(['subject','session','epoch_id']).ngroups}")

    df["global_epoch"] = (df["subject"] + "_" +
                          df["session"].astype(str) + "_" +
                          df["label"].astype(str) + "_" +
                          df["epoch_id"].astype(str))

    features_list    = []
    plotted_subjects = set()

    print(f"\nEstrazione features con NeuroKit2 (window={WINDOW_SEC}s) ...")

    for global_id, epoch_data in df.groupby("global_epoch"):
        subject  = epoch_data["subject"].iloc[0]
        session  = epoch_data["session"].iloc[0]
        label    = epoch_data["label"].iloc[0]
        epoch_id = epoch_data["epoch_id"].iloc[0]

        epoch_features = {
            "subject":  subject,
            "session":  session,
            "epoch_id": epoch_id,
            "label":    label,
        }

        psd_store = {}

        for ch in CHANNELS:
            sig = epoch_data[ch].values
            
            # 1. Calcolo Spettro per i grafici di Sanity Check
            psd_store[ch] = nk.signal_psd(sig, sampling_rate=SFREQ, method="welch", show=False, window=WINDOW_SEC)

            # 2. Estrazione Potenza Assoluta nelle singole bande
            p_theta = nk.signal_power(sig, frequency_band=[(4.0, 8.0)], sampling_rate=SFREQ, method="welch", normalize=False, window=WINDOW_SEC).iloc[0, 0]
            p_alpha = nk.signal_power(sig, frequency_band=[(8.0, 13.0)], sampling_rate=SFREQ, method="welch", normalize=False, window=WINDOW_SEC).iloc[0, 0]
            p_beta  = nk.signal_power(sig, frequency_band=[(13.0, 30.0)], sampling_rate=SFREQ, method="welch", normalize=False, window=WINDOW_SEC).iloc[0, 0]
            p_total = nk.signal_power(sig, frequency_band=[(1.0, 40.0)], sampling_rate=SFREQ, method="welch", normalize=False, window=WINDOW_SEC).iloc[0, 0]

            # 3. Calcolo metriche relative e rapporti
            rel_theta = p_theta / p_total if p_total > 0 else 0
            rel_alpha = p_alpha / p_total if p_total > 0 else 0
            rel_beta  = p_beta  / p_total if p_total > 0 else 0

            tbr = p_theta / p_beta  if p_beta  > 0 else 0
            eng = p_beta  / p_alpha if p_alpha > 0 else 0

            # 4. Salvataggio in memoria
            epoch_features[f"Abs_Theta_{ch}"] = p_theta
            epoch_features[f"Abs_Alpha_{ch}"] = p_alpha
            epoch_features[f"Abs_Beta_{ch}"]  = p_beta
            epoch_features[f"Rel_Theta_{ch}"] = rel_theta
            epoch_features[f"Rel_Alpha_{ch}"] = rel_alpha
            epoch_features[f"Rel_Beta_{ch}"]  = rel_beta
            epoch_features[f"TBR_{ch}"]       = tbr
            epoch_features[f"Eng_{ch}"]       = eng

        # Frontal Alpha Asymmetry
        a_af8 = epoch_features["Abs_Alpha_AF8"]
        a_af7 = epoch_features["Abs_Alpha_AF7"]
        epoch_features["FAA"] = (np.log(a_af8) - np.log(a_af7)
                                  if a_af8 > 0 and a_af7 > 0 else 0)

        features_list.append(epoch_features)

        # Sanity check: salva immagine per la prima epoca neutral/concentrating di ogni soggetto
        if subject not in plotted_subjects and label == 0:
            save_sanity_check(psd_store, subject, "neutral")
            plotted_subjects.add(subject)

    df_features = pd.DataFrame(features_list)
    df_features.to_csv(OUTPUT_CSV, index=False)

    n_feat = len([c for c in df_features.columns
                  if c not in ("subject", "session", "epoch_id", "label")])

    print(f"\nCompletato!")
    print(f"  Epoche estratte : {len(df_features)}")
    print(f"  Feature per epoca: {n_feat}")
    print(f"  File salvato    : {OUTPUT_CSV}")
    print(f"\nDistribuzione label:")
    print(df_features["label"].value_counts().rename({0: "neutral", 1: "concentrating"}).to_string())
    print(f"\nEpoche per soggetto/sessione:")
    print(df_features.groupby(["subject", "session", "label"])["epoch_id"]
          .count().rename("epoche").to_string())

if __name__ == "__main__":
    main()