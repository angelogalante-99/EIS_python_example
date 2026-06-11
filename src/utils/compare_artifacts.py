"""
Confronto Artefatti — Dominio Tempo vs Frequenza
-------------------------------------------------
Genera una figura a 4 pannelli per confrontare visivamente
un soggetto "pulito" (subjecta) vs uno con EMG (subjectb):

  Riga 1 — Dominio del TEMPO : un'epoca rappresentativa per soggetto
  Riga 2 — Dominio della FREQUENZA : PSD media su tutte le epoche

Cosa cercare:
  Tempo   → "pelo" fitto e caotico su un canale = EMG
  Freq    → curva piatta in beta (13-30 Hz) invece di decadere = EMG

Uso:
    python compare_artifacts.py
"""

import os
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

# ── Configurazione ───────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH  = os.path.join(BASE_DIR, "data", "eeg_dataset_preprocessed.csv")
OUT_PATH  = os.path.join(BASE_DIR, "src", "features", "sanity_checks",
                         "compare_artifacts.png")

SFREQ      = 256.0
WINDOW_SEC = 1
CHANNELS   = ["TP9", "AF7", "AF8", "TP10"]

# Soggetti da confrontare — cambia qui se vuoi altri soggetti
SUBJECT_CLEAN   = "subjecta"   # segnale pulito
SUBJECT_NOISY   = "subjectc"   # EMG sui canali temporali

# Stato da visualizzare — di solito concentrating mostra più artefatti
LABEL           = 1            # 0=neutral  1=concentrating

CH_COLORS = {"TP9": "#185FA5", "AF7": "#D85A30",
             "AF8": "#1D9E75", "TP10": "#854F0B"}
BANDS     = [(4, 8,  "blue",  0.07, "θ 4-8"),
             (8, 13, "green", 0.15, "α 8-13"),
             (13, 30,"red",   0.07, "β 13-30")]
LABEL_STR = {0: "neutral", 1: "concentrating"}
# ─────────────────────────────────────────────────────────────────────────────


def get_epoch(df, subject, label, epoch_index=0):
    """
    Estrae una singola epoca.
    epoch_index: quale epoca prendere (0 = prima disponibile).
    Cambialo se la prima è troppo rumorosa o troppo piatta —
    prova 1, 2, 3 per trovare un esempio rappresentativo.
    """
    sub = df[(df["subject"] == subject) & (df["label"] == label)]
    epoch_ids = sorted(sub["epoch_id"].unique())
    if epoch_index >= len(epoch_ids):
        epoch_index = 0
    eid = epoch_ids[epoch_index]
    return sub[sub["epoch_id"] == eid]


def compute_mean_psd(df, subject, label):
    """Calcola la PSD media su tutte le epoche del soggetto/stato."""
    sub    = df[(df["subject"] == subject) & (df["label"] == label)]
    accum  = defaultdict(list)
    n      = 0
    for _, ep in sub.groupby(["session", "epoch_id"]):
        for ch in CHANNELS:
            psd_df = nk.signal_psd(ep[ch].values, sampling_rate=SFREQ,
                                   method="welch", show=False, window=WINDOW_SEC)
            psd_df = psd_df[psd_df["Frequency"] <= 40]
            accum[ch].append(psd_df.set_index("Frequency")["Power"])
        n += 1
    mean_psds = {}
    for ch in CHANNELS:
        stacked      = pd.concat(accum[ch], axis=1).ffill()
        mean_psds[ch]= stacked.mean(axis=1)
    return mean_psds, n


def plot_time(ax, epoch_df, subject, label, epoch_index):
    """Pannello dominio del tempo: tracce µV vs campioni."""
    t = np.arange(len(epoch_df)) / SFREQ   # asse x in secondi

    # Offset verticale per separare i canali visivamente (come fa MNE)
    offset   = 60   # µV tra un canale e l'altro — abbassa se si sovrappongono
    n_ch     = len(CHANNELS)

    for i, ch in enumerate(CHANNELS):
        sig   = epoch_df[ch].values
        y_off = (n_ch - 1 - i) * offset
        ax.plot(t, sig + y_off, color=CH_COLORS[ch], linewidth=0.7,
                alpha=0.9, label=ch)
        ax.axhline(y_off, color="gray", linewidth=0.3, linestyle="--", alpha=0.4)

    ax.set_xlim(0, t[-1])
    ax.set_xlabel("Tempo (s)", fontsize=9)
    ax.set_ylabel("Ampiezza (µV) + offset", fontsize=9)
    ax.set_title(
        f"{subject} | {LABEL_STR[label]} | epoca #{epoch_index}\n"
        f"← Dominio del TEMPO: cerca 'pelo fitto' su un canale = EMG",
        fontsize=9, loc="left"
    )
    ax.set_yticks([(n_ch - 1 - i) * offset for i in range(n_ch)])
    ax.set_yticklabels(CHANNELS, fontsize=8)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")


def plot_freq(ax, mean_psds, subject, label, n_epochs):
    """Pannello dominio della frequenza: PSD media in dB/Hz."""
    for lo, hi, col, alpha, name in BANDS:
        ax.axvspan(lo, hi, color=col, alpha=alpha, label=name)

    for ch in CHANNELS:
        psd_db = 10 * np.log10(np.maximum(mean_psds[ch].values, 1e-20))
        ax.plot(mean_psds[ch].index, psd_db,
                color=CH_COLORS[ch], linewidth=1.6, alpha=0.9, label=ch)

    ax.set_xlim(1, 40)
    ax.set_xlabel("Frequenza (Hz)  ←  non è una durata!", fontsize=9)
    ax.set_ylabel("Potenza Spettrale (dB/Hz)", fontsize=9)
    ax.set_title(
        f"{subject} | {LABEL_STR[label]} | media {n_epochs} epoche\n"
        f"← Dominio della FREQUENZA: curva piatta in β = EMG",
        fontsize=9, loc="left"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=7, loc="upper right")


def main():
    print(f"Caricamento {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    # ── Quale epoca mostrare nel plot temporale ───────────────────────────────
    # epoch_index=0 → prima epoca disponibile.
    # Se l'artefatto non si vede bene, prova 1, 2, 3...
    EPOCH_IDX_CLEAN = 10
    EPOCH_IDX_NOISY = 10  # per subjectb concentrating anche la prima è già evidente

    print(f"Estrazione epoche ...")
    ep_clean = get_epoch(df, SUBJECT_CLEAN, LABEL, EPOCH_IDX_CLEAN)
    ep_noisy = get_epoch(df, SUBJECT_NOISY, LABEL, EPOCH_IDX_NOISY)

    print(f"Calcolo PSD media ...")
    psd_clean, n_clean = compute_mean_psd(df, SUBJECT_CLEAN, LABEL)
    psd_noisy, n_noisy = compute_mean_psd(df, SUBJECT_NOISY, LABEL)

    # ── Figura 2×2 ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Confronto artefatti — {LABEL_STR[LABEL]}\n"
        f"Sinistra: {SUBJECT_CLEAN} (segnale pulito)   |   "
        f"Destra: {SUBJECT_NOISY} (EMG sui canali temporali)",
        fontsize=11, y=1.01
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax_t_clean = fig.add_subplot(gs[0, 0])
    ax_t_noisy = fig.add_subplot(gs[0, 1])
    ax_f_clean = fig.add_subplot(gs[1, 0])
    ax_f_noisy = fig.add_subplot(gs[1, 1])

    # Riga 0 — tempo
    plot_time(ax_t_clean, ep_clean, SUBJECT_CLEAN, LABEL, EPOCH_IDX_CLEAN)
    plot_time(ax_t_noisy, ep_noisy, SUBJECT_NOISY, LABEL, EPOCH_IDX_NOISY)

    # Riga 1 — frequenza
    plot_freq(ax_f_clean, psd_clean, SUBJECT_CLEAN, LABEL, n_clean)
    plot_freq(ax_f_noisy, psd_noisy, SUBJECT_NOISY, LABEL, n_noisy)

    # Allinea assi Y della PSD per confronto diretto
    y_min = min(ax_f_clean.get_ylim()[0], ax_f_noisy.get_ylim()[0])
    y_max = max(ax_f_clean.get_ylim()[1], ax_f_noisy.get_ylim()[1])
    ax_f_clean.set_ylim(y_min, y_max)
    ax_f_noisy.set_ylim(y_min, y_max)

    # Annotazioni guida
    ax_t_noisy.annotate(
        "← 'pelo' fitto = EMG",
        xy=(0.5, 0.5), xycoords="axes fraction",
        fontsize=8, color="#D85A30", alpha=0.7,
        ha="center", style="italic"
    )
    ax_f_noisy.annotate(
        "← piatto in β = EMG",
        xy=(0.65, 0.55), xycoords="axes fraction",
        fontsize=8, color="#D85A30", alpha=0.7,
        ha="center", style="italic"
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    plt.savefig(OUT_PATH, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"\nFigura salvata: {OUT_PATH}")
    print("\nCosa guardare:")
    print("  TEMPO   — se vedi oscillazioni fitte e caotiche su TP9/TP10 in subjectb = EMG")
    print("  FREQ    — se la curva si appiattisce nella zona rossa (13-30 Hz) = EMG")
    print("  CONFRONTO — subjecta dovrebbe avere curve che decadono; subjectb no")
    print("\nSe l'epoca scelta non è rappresentativa, cambia EPOCH_IDX_NOISY (prova 1,2,3)")


if __name__ == "__main__":
    main()