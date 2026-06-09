"""
Data Extraction — EEG Dataset
------------------------------
Legge i CSV dall'archivio ZIP e li unisce in un unico file raw.

Le sessioni vengono numerate in modo UNIVOCO per soggetto:
  subjecta-neutral-1.csv       → session=1, label=0
  subjecta-neutral-2.csv       → session=2, label=0
  subjecta-concentrating-1.csv → session=3, label=1
  subjecta-concentrating-2.csv → session=4, label=1

Garanzia: ogni (subject, session) ha sempre una sola label.
"""

import zipfile
import pandas as pd
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH    = os.path.join(BASE_DIR, "original_data.zip")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "eeg_dataset_raw.csv")

dfs = []
session_counter = {}  # {subject: contatore sessione univoco}

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    # Ordine alfabetico garantisce: concentrating-1, concentrating-2, neutral-1, neutral-2
    csv_files = sorted([f for f in z.namelist() if f.endswith(".csv")])

    for filename in csv_files:
        basename = os.path.basename(filename)
        parts    = basename.replace(".csv", "").split("-")

        if len(parts) < 3:
            print(f"  Ignorato: {filename}")
            continue

        subject = parts[0]
        state   = parts[1]
        session = int(parts[2])

        if state == "relaxed":
            continue

        label = 1 if state == "concentrating" else 0

        # Incrementa contatore univoco per soggetto
        if subject not in session_counter:
            session_counter[subject] = 0
        session_counter[subject] += 1
        unique_session = session_counter[subject]

        with z.open(filename) as f:
            df = pd.read_csv(f)

        df["subject"] = subject
        df["session"] = unique_session
        df["label"]   = label

        dfs.append(df)
        print(f"  {basename:45s} → session={unique_session}  label={label}  ({len(df):,} righe)")

merged = pd.concat(dfs, ignore_index=True)

signal_cols = [c for c in merged.columns if c not in ["subject", "session", "label"]]
merged      = merged[["subject", "session"] + signal_cols + ["label"]]

# Verifica finale
check    = merged.groupby(["subject", "session"])["label"].nunique()
problemi = check[check > 1]
if problemi.empty:
    print("\nVerifica OK — ogni (subject, session) ha una sola label.")
else:
    print("\nERRORE — sessioni con label multiple:")
    print(problemi)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
merged.to_csv(OUTPUT_PATH, index=False)

print(f"\nSalvato: {OUTPUT_PATH}")
print(f"Righe totali: {len(merged):,}")
print(f"\nDistribuzione (subject, session, label):")
print(merged.groupby(["subject", "session", "label"])["timestamps"]
      .count().rename("campioni").to_string())
