"""
EEG Feature Extraction Pipeline — shortlist neurale (6 feature)
----------------------------------------------------------------
Estrae solo le feature che hanno mostrato:
  1. Effect size (Cohen's d) coerente in direzione su tutti e 4 i soggetti
     (neutral vs concentrating, calcolato per soggetto)
  2. Nessuna contaminazione EMG accertata (banda < 13 Hz o rapporto
     che cancella la scala assoluta)
  3. Localizzazione su canali frontali (AF7, AF8), gli unici che
     mostrano dinamica cognitiva sul Muse a 4 canali

Feature scartate e motivazione:
  - Abs/Rel_Beta_*, TBR_*, Eng_*  : banda 13-30 Hz sovrapposta all'EMG
    muscolare (mascella, fronte). Analisi effect-size mostrava d incoerente
    tra soggetti (cambio di segno), segno che la separazione era artefatto
    specifico del soggetto, non segnale neurale.
  - Tutti i canali TP9/TP10        : canali temporali posizionati sopra i
    muscoli masticatori. PSD media mostrava gradino piatto in beta per
    subjectb (EMG temporale confermato). Effect size massimo 0.23 — troppo
    basso per contribuire in modo affidabile.
  - Abs/Rel_Alpha_AF7, Abs/Rel_Theta_AF8 solo incluse dove d > 0.5 e
    coerenti (vedi tabella nell'analisi).

Feature mantenute:
  - Rel_Alpha_AF8 : soppressione alpha relativa su AF8 (frontale dx).
                    Marcatore neurale classico dell'attenzione focalizzata;
                    d medio pesato = 1.93, coerente su tutti e 4 i soggetti.
  - Abs_Alpha_AF8 : stessa dinamica in scala assoluta; utile per modelli
                    non normalizzati. d = 1.38, coerente 4/4.
  - Abs_Theta_AF8 : aumento del theta frontale associato al carico cognitivo
                    (working memory). d = 1.13, coerente 4/4.
  - Rel_Theta_AF7 : theta relativa su AF7 (frontale sx); conferma la
                    dinamica theta su entrambi i canali frontali. d = 0.93.
  - Abs_Theta_AF7 : theta assoluta AF7; complementare a Rel per robustezza
                    rispetto a variazioni di scala inter-soggetto. d = 0.55.
  - FAA           : Frontal Alpha Asymmetry = log(alpha_AF8) - log(alpha_AF7).
                    Indicatore di asimmetria emisferico-frontale dell'alpha;
                    d = 1.50, coerente 4/4. Cattura differenze laterali che
                    le singole feature per canale non vedono.

Sanity check: al termine salva PSD media per soggetto × stato (neutral e
concentrating) per consentire verifica visiva della qualità del segnale.
"""

import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import os
from collections import defaultdict

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_CSV  = os.path.join(BASE_DIR, "data", "eeg_dataset_preprocessed.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "data", "final_features.csv")
SANITY_DIR = os.path.join(BASE_DIR, "src", "features", "sanity_checks")

SFREQ      = 256.0
CHANNELS   = ["TP9", "AF7", "AF8", "TP10"]
WINDOW_SEC = 1   # finestre Welch da 1 s → risoluzione spettrale 1 Hz

# Nomi delle 6 feature, nello stesso ordine restituito da extract_features()
# e nello stesso ordine delle colonne di final_features.csv
FEATURE_NAMES = [
    "Abs_Theta_AF7", "Rel_Theta_AF7",
    "Abs_Alpha_AF8", "Rel_Alpha_AF8",
    "Abs_Theta_AF8", "FAA",
]


def band_power(psd_df, lo, hi):
    """Integra una PSD (DataFrame con colonne Frequency/Power) su [lo, hi) Hz."""
    f = psd_df["Frequency"].values
    p = psd_df["Power"].values
    mask = (f >= lo) & (f < hi)
    return np.trapezoid(p[mask], f[mask]) if mask.sum() > 1 else 0.0


def extract_features(epoch_signals, sfreq=SFREQ, window_sec=WINDOW_SEC):
    """
    Estrae le 6 feature della shortlist neurale da un'epoca.

    epoch_signals: dict {"AF7": array_1d, "AF8": array_1d}
                   (gli unici due canali usati dalle 6 feature)
    Ritorna: dict con le 6 feature, chiavi = FEATURE_NAMES

    Questa è l'UNICA implementazione della logica di estrazione:
    usata sia dal batch (main() qui sotto, su tutto il dataset) sia
    da prediction.py (in tempo reale, su un'epoca dallo stream).
    """
    psd_af7 = nk.signal_psd(epoch_signals["AF7"], sampling_rate=sfreq,
                            method="welch", show=False, window=window_sec)
    psd_af8 = nk.signal_psd(epoch_signals["AF8"], sampling_rate=sfreq,
                            method="welch", show=False, window=window_sec)

    p_total_af7 = band_power(psd_af7, 1.0, 40.0)
    p_total_af8 = band_power(psd_af8, 1.0, 40.0)

    # Theta AF7 (4-8 Hz) — coerente su 4/4 soggetti, d medio pesato = 0.93
    p_theta_af7 = band_power(psd_af7, 4.0, 8.0)
    abs_theta_af7 = p_theta_af7
    rel_theta_af7 = p_theta_af7 / p_total_af7 if p_total_af7 > 0 else 0.0

    # Alpha AF8 (8-13 Hz) — soppressione alpha, feature più discriminante
    # d = 1.93 (Rel) e 1.38 (Abs), coerente 4/4
    p_alpha_af8 = band_power(psd_af8, 8.0, 13.0)
    abs_alpha_af8 = p_alpha_af8
    rel_alpha_af8 = p_alpha_af8 / p_total_af8 if p_total_af8 > 0 else 0.0

    # Theta AF8 (4-8 Hz) — d = 1.13, coerente 4/4
    p_theta_af8 = band_power(psd_af8, 4.0, 8.0)
    abs_theta_af8 = p_theta_af8

    # FAA = log(alpha_AF8) - log(alpha_AF7) — d = 1.50, coerente 4/4
    p_alpha_af7 = band_power(psd_af7, 8.0, 13.0)
    faa = (np.log(p_alpha_af8) - np.log(p_alpha_af7)
           if p_alpha_af8 > 0 and p_alpha_af7 > 0 else 0.0)

    return {
        "Abs_Theta_AF7": abs_theta_af7,
        "Rel_Theta_AF7": rel_theta_af7,
        "Abs_Alpha_AF8": abs_alpha_af8,
        "Rel_Alpha_AF8": rel_alpha_af8,
        "Abs_Theta_AF8": abs_theta_af8,
        "FAA":           faa,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Sanity check PSD medio
# ──────────────────────────────────────────────────────────────────────────────

def save_mean_sanity_check(mean_psds, subject, state, n_epochs, out_dir=SANITY_DIR):
    """
    Salva il grafico PSD medio (media su tutte le epoche dello stato) su disco.

    mean_psds : dict  {ch: Series con index=Frequency, values=Power (lineare)}
    n_epochs  : int   numero di epoche usate per la media
    """
    os.makedirs(out_dir, exist_ok=True)
    subj_short = subject.split("/")[-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    for ch in CHANNELS:
        if ch not in mean_psds:
            continue
        s = mean_psds[ch]
        s = s[s.index <= 40.0]
        ax.plot(s.index, 10 * np.log10(np.maximum(s.values, 1e-20)),
                label=ch, alpha=0.85)

    ax.axvspan(4,  8,  color="blue",  alpha=0.07, label="Theta (4-8 Hz)")
    ax.axvspan(8,  13, color="green", alpha=0.15, label="Alpha (8-13 Hz)")
    ax.axvspan(13, 30, color="red",   alpha=0.07, label="Beta  (13-30 Hz)")

    label_str = "neutral" if state == 0 else "concentrating"
    ax.set_title(f"Sanity Check PSD (media {n_epochs} epoche) — {subj_short} | {label_str}")
    ax.set_xlabel("Frequenza (Hz)")
    ax.set_ylabel("Potenza Spettrale (dB/Hz)")
    ax.set_xlim(1, 40)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{subj_short}_{label_str}.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  Sanity check salvato ({n_epochs} epoche): {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

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

    features_list = []
    psd_accum     = defaultdict(lambda: defaultdict(list))

    print(f"\nEstrazione features (window={WINDOW_SEC}s) ...")

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

        # PSD per il sanity check (tutti e 4 i canali, media per soggetto/stato)
        for ch in CHANNELS:
            sig    = epoch_data[ch].values
            psd_df = nk.signal_psd(sig, sampling_rate=SFREQ, method="welch",
                                   show=False, window=WINDOW_SEC)
            psd_accum[(subject, label)][ch].append(
                psd_df.set_index("Frequency")["Power"]
            )

        # Le 6 feature della shortlist — stessa funzione usata da prediction.py
        epoch_signals = {ch: epoch_data[ch].values for ch in ["AF7", "AF8"]}
        epoch_features.update(extract_features(epoch_signals))

        features_list.append(epoch_features)

    # ── Salvataggio CSV ───────────────────────────────────────────────────────
    df_features = pd.DataFrame(features_list)
    df_features.to_csv(OUTPUT_CSV, index=False)

    feat_cols = [c for c in df_features.columns if c not in ("subject", "session", "epoch_id", "label")]
    print(f"\nCompletato!")
    print(f"  Epoche estratte  : {len(df_features)}")
    print(f"  Feature per epoca: {len(feat_cols)}  →  {feat_cols}")
    print(f"  File salvato     : {OUTPUT_CSV}")
    print(f"\nDistribuzione label:")
    print(df_features["label"].value_counts().rename({0: "neutral", 1: "concentrating"}).to_string())
    print(f"\nEpoche per soggetto/sessione:")
    print(df_features.groupby(["subject", "session", "label"])["epoch_id"]
          .count().rename("epoche").to_string())

    # ── Sanity check medi ─────────────────────────────────────────────────────
    print(f"\nGenerazione sanity check medi ...")
    for (subject, label), ch_lists in sorted(psd_accum.items()):
        mean_psds = {}
        for ch, series_list in ch_lists.items():
            stacked = pd.concat(series_list, axis=1).ffill()
            mean_psds[ch] = stacked.mean(axis=1)
        n_epochs = len(next(iter(ch_lists.values())))
        save_mean_sanity_check(mean_psds, subject, label, n_epochs)


if __name__ == "__main__":
    main()