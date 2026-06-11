"""
Analisi Feature — Cohen's d su tutti i canali
----------------------------------------------
Calcola per ogni feature e ogni soggetto il Cohen's d
(neutral vs concentrating) e plotta:
  1. Tabella heatmap dei d per soggetto
  2. Grafico a barre del |d| medio pesato
  3. Stampa a console il calcolo passo per passo

Feature calcolate (26 totali):
  - Abs/Rel Theta (4-8 Hz)   per TP9, AF7, AF8, TP10  → 8 feature
  - Abs/Rel Alpha (8-13 Hz)  per TP9, AF7, AF8, TP10  → 8 feature
  - Abs/Rel Beta  (13-30 Hz) per TP9, AF7, AF8, TP10  → 8 feature
  - FAA  (log alpha AF8 - log alpha AF7)               → 1 feature
  - TBR_AF7, TBR_AF8  (theta/beta ratio)               → 2 feature

Uso:
    python feature_analysis.py
"""

import os
import pandas as pd
import numpy as np
import neurokit2 as nk
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict

# ── Configurazione ───────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data", "eeg_dataset_preprocessed.csv")
OUT_DIR  = os.path.join(BASE_DIR, "src", "features", "sanity_checks")

SFREQ      = 256.0
WINDOW_SEC = 1
CHANNELS   = ["TP9", "AF7", "AF8", "TP10"]
# subjectc escluso: elettrodo frontale compromesso in tutte le sessioni
# (AF7 rotta nelle sessioni concentrating, AF8 rotta nelle sessioni neutral)
# confermato da ispezione visiva del segnale e analisi PSD media per sessione
SUBJECTS = ["subjecta", "subjectb", "subjectd"]

# Peso pulizia data-driven (inverso indice EMG 30-40 Hz), rinormalizzato su 3 soggetti
WEIGHTS = {"subjecta": 0.364, "subjectb": 0.401, "subjectd": 0.236}

# Bande di frequenza
BANDS = {"Theta": (4.0, 8.0), "Alpha": (8.0, 13.0), "Beta": (13.0, 30.0)}
# ─────────────────────────────────────────────────────────────────────────────


def band_power(sig, lo, hi, sfreq=SFREQ, window=WINDOW_SEC):
    """Integra la PSD Welch nella banda [lo, hi) Hz. Ritorna potenza assoluta."""
    psd_df = nk.signal_psd(sig, sampling_rate=sfreq, method="welch",
                           show=False, window=window)
    f = psd_df["Frequency"].values
    p = psd_df["Power"].values
    mask = (f >= lo) & (f < hi)
    return np.trapezoid(p[mask], f[mask]) if mask.sum() > 1 else 0.0


def extract_all_features(df):
    """
    Estrae tutte le 26 feature per ogni epoca.
    Ritorna un DataFrame con colonne: subject, label, + una per feature.
    """
    df["gid"] = (df["subject"] + "_" + df["session"].astype(str) + "_" +
                 df["label"].astype(str) + "_" + df["epoch_id"].astype(str))

    rows = []
    total = df["gid"].nunique()
    print(f"  Calcolo feature su {total} epoche ...")

    for i, (gid, ep) in enumerate(df.groupby("gid")):
        if i % 100 == 0:
            print(f"    {i}/{total}", end="\r")

        row = {"subject": ep["subject"].iloc[0],
               "label":   ep["label"].iloc[0]}

        bp = {}   # bp[ch][band] = potenza assoluta
        for ch in CHANNELS:
            sig       = ep[ch].values
            p_total   = band_power(sig, 1.0, 40.0)
            bp[ch]    = {}
            for band, (lo, hi) in BANDS.items():
                pa = band_power(sig, lo, hi)
                bp[ch][band] = pa
                row[f"Abs_{band}_{ch}"] = pa
                row[f"Rel_{band}_{ch}"] = pa / p_total if p_total > 0 else 0.0

        # FAA — Frontal Alpha Asymmetry
        a8 = bp["AF8"]["Alpha"]
        a7 = bp["AF7"]["Alpha"]
        row["FAA"] = np.log(a8) - np.log(a7) if a8 > 0 and a7 > 0 else 0.0

        # TBR — Theta/Beta Ratio (solo frontali, usato in letteratura concentrazione)
        for ch in ["AF7", "AF8"]:
            t = bp[ch]["Theta"]
            b = bp[ch]["Beta"]
            row[f"TBR_{ch}"] = t / b if b > 0 else 0.0

        rows.append(row)

    print(f"    {total}/{total} — completato")
    return pd.DataFrame(rows)


def cohend(neutral, concentrating):
    """
    Calcola Cohen's d con tutti i passaggi intermedi.
    Ritorna (d, dizionario con i passi per stampa).
    """
    n1, n2 = len(neutral), len(concentrating)
    mn, mc = neutral.mean(), concentrating.mean()
    vn, vc = neutral.var(ddof=1), concentrating.var(ddof=1)

    # Std pooled: media pesata delle varianze per gradi di libertà
    sp = np.sqrt(((n1-1)*vn + (n2-1)*vc) / (n1+n2-2))
    d  = (mc - mn) / sp if sp > 0 else 0.0

    steps = {"n_neutral": n1, "n_conc": n2,
             "mean_n": mn, "mean_c": mc,
             "std_n": np.sqrt(vn), "std_c": np.sqrt(vc),
             "sp": sp, "d": d}
    return d, steps


def compute_d_table(feat_df):
    """
    Calcola d per ogni (feature × soggetto) e il |d| medio pesato.
    Ritorna DataFrame con index=feature, colonne=soggetti + weighted_abs_d.
    """
    feat_cols = [c for c in feat_df.columns if c not in ("subject", "label")]
    records   = []

    for feat in feat_cols:
        row = {"feature": feat}
        d_vals = {}
        for subj in SUBJECTS:
            sub = feat_df[feat_df["subject"] == subj]
            n_  = sub[sub["label"] == 0][feat].values
            c_  = sub[sub["label"] == 1][feat].values
            d, _ = cohend(n_, c_)
            d_vals[subj] = d
            row[f"d_{subj[-1]}"] = round(d, 3)

        # |d| medio pesato per i pesi di pulizia
        w_abs = sum(WEIGHTS[s] * abs(d_vals[s]) for s in SUBJECTS)
        row["w_abs_d"] = round(w_abs, 3)

        # Coerenza di direzione (tutti stesso segno?)
        signs = [np.sign(d_vals[s]) for s in SUBJECTS]
        row["coerente"] = "SI ✓" if len(set(signs)) == 1 else "NO ✗"

        # Flag EMG sospetto (beta o engagement su qualsiasi canale)
        row["EMG_susp"] = "Beta" in feat or "TBR" in feat

        records.append(row)

    return pd.DataFrame(records).sort_values("w_abs_d", ascending=False)


def print_steps(feat_df, feature, subject):
    """Stampa il calcolo passo per passo per una feature/soggetto specifici."""
    sub = feat_df[feat_df["subject"] == subject]
    n_  = sub[sub["label"] == 0][feature].values
    c_  = sub[sub["label"] == 1][feature].values
    d, s = cohend(n_, c_)

    print(f"\n{'='*55}")
    print(f"  Cohen's d — {feature} — {subject}")
    print(f"{'='*55}")
    print(f"  Epoche neutral       : {s['n_neutral']}")
    print(f"  Epoche concentrating : {s['n_conc']}")
    print(f"  media neutral        : {s['mean_n']:.5f}  (std={s['std_n']:.5f})")
    print(f"  media concentrating  : {s['mean_c']:.5f}  (std={s['std_c']:.5f})")
    print(f"  differenza grezza    : {s['mean_c']-s['mean_n']:+.5f}")
    print(f"  std pooled           : {s['sp']:.5f}")
    print(f"  Cohen's d            : {d:+.4f}")
    dir_str = "SCENDE" if d < 0 else "SALE"
    print(f"  → {feature} {dir_str} di {abs(d):.2f} std durante concentrating")


def plot_heatmap(d_table):
    """
    Heatmap: righe = feature (ordinate per |d| pesato),
             colonne = soggetti.
    Colori: blu = scende in concentrating, rosso = sale.
    """
    feats   = d_table["feature"].tolist()
    d_cols  = [f"d_{s[-1]}" for s in SUBJECTS]
    matrix  = d_table[d_cols].values.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, max(8, len(feats)*0.38)),
                             gridspec_kw={"width_ratios": [3, 1]})

    # ── Heatmap ──────────────────────────────────────────────
    ax = axes[0]
    vmax = np.nanmax(np.abs(matrix))
    im   = ax.imshow(matrix, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(SUBJECTS)))
    ax.set_xticklabels([s[-1].upper() for s in SUBJECTS], fontsize=11)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8.5, fontfamily="monospace")
    ax.set_xlabel("Soggetto", fontsize=10)
    ax.set_title("Cohen's d per feature × soggetto\n"
                 "blu = scende in concentrating  |  rosso = sale", fontsize=10)

    # Valori numerici nelle celle
    for i in range(len(feats)):
        for j in range(len(SUBJECTS)):
            v = matrix[i, j]
            color = "white" if abs(v) > vmax * 0.6 else "black"
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                    fontsize=7.5, color=color)

    # Linee separatrici tra gruppi di feature (theta / alpha / beta / cross)
    separators = []
    prev_band  = None
    for i, f in enumerate(feats):
        band = ("Alpha" if "Alpha" in f else "Theta" if "Theta" in f
                else "Beta" if "Beta" in f else "Cross")
        if prev_band and band != prev_band:
            separators.append(i - 0.5)
        prev_band = band
    for s in separators:
        ax.axhline(s, color="white", linewidth=1.5, alpha=0.7)

    # Evidenzia righe con coerenza SI
    for i, row in d_table.iterrows():
        idx = feats.index(row["feature"])
        if row["coerente"] == "SI ✓":
            ax.add_patch(plt.Rectangle((-0.5, idx-0.5), len(SUBJECTS), 1,
                                       fill=False, edgecolor="#1D9E75",
                                       linewidth=1.5))

    plt.colorbar(im, ax=ax, label="Cohen's d", fraction=0.03)

    # ── Barre |d| medio pesato ───────────────────────────────
    ax2 = axes[1]
    colors = ["#D85A30" if row["EMG_susp"] else
              "#1D9E75" if row["coerente"] == "SI ✓" else "#7F77DD"
              for _, row in d_table.iterrows()]
    ax2.barh(range(len(feats)), d_table["w_abs_d"].values,
             color=colors, alpha=0.8, height=0.6)
    ax2.set_yticks(range(len(feats)))
    ax2.set_yticklabels([])
    ax2.set_xlabel("|d| medio pesato", fontsize=9)
    ax2.set_title("Discriminabilità\n(pesata per pulizia)", fontsize=9)
    ax2.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.axvline(1.0, color="gray", linestyle="-",  linewidth=0.8, alpha=0.6)
    ax2.text(0.5, len(feats)+0.3, "0.5", fontsize=7, ha="center", color="gray")
    ax2.text(1.0, len(feats)+0.3, "1.0", fontsize=7, ha="center", color="gray")

    # Legenda colori barre
    from matplotlib.patches import Patch
    legend = [Patch(color="#1D9E75", label="coerente + non EMG"),
              Patch(color="#7F77DD", label="incoerente + non EMG"),
              Patch(color="#D85A30", label="sospetto EMG (beta/TBR)")]
    ax2.legend(handles=legend, fontsize=7, loc="lower right")

    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "feature_cohend_heatmap.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.show()
    print(f"\n  Heatmap salvata: {out_path}")


def main():
    print(f"Caricamento {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)

    print("\nEstrazione feature (26 per epoca) ...")
    feat_df = extract_all_features(df)

    print("\nCalcolo Cohen's d ...")
    d_table = compute_d_table(feat_df)

    # ── Stampa tabella ordinata ──────────────────────────────
    print("\n" + "="*75)
    print("  TABELLA Cohen's d — ordinata per |d| medio pesato")
    print("="*75)
    subj_headers = "  ".join(f"d_{s[-1].upper()}" for s in SUBJECTS)
    print(f"  {'feature':<20} {subj_headers}  {'w|d|':>6}  {'coer':>6}  {'EMG?':>5}")
    print("  " + "-"*70)
    for _, row in d_table.iterrows():
        emg  = "⚠" if row["EMG_susp"] else " "
        d_vals = "  ".join(f"{row[f'd_{s[-1]}']:>+6.2f}" for s in SUBJECTS)
        print(f"  {row['feature']:<20} {d_vals}  "
              f"{row['w_abs_d']:>6.2f}  {row['coerente']:>6}  {emg:>5}")

    # ── Stampa calcolo passo per passo per le top 3 ─────────
    print("\n\n--- Calcolo passo per passo — top 3 feature ---")
    for feat in d_table.head(3)["feature"].tolist():
        print_steps(feat_df, feat, "subjecta")

    # ── Plot ─────────────────────────────────────────────────
    print("\nGenerazione heatmap ...")
    plot_heatmap(d_table)

    # ── Shortlist consigliata ────────────────────────────────
    print("\n" + "="*75)
    print("  SHORTLIST CONSIGLIATA")
    print("  criteri: coerente su 4/4 soggetti + non EMG-sospetta + w|d| > 0.4")
    print("="*75)
    shortlist = d_table[
        (d_table["coerente"] == "SI ✓") &
        (~d_table["EMG_susp"]) &
        (d_table["w_abs_d"] > 0.4)
    ]
    for _, row in shortlist.iterrows():
        print(f"  {row['feature']:<22}  w|d|={row['w_abs_d']:.2f}")


if __name__ == "__main__":
    main()