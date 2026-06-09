import zipfile
import pandas as pd

ZIP_PATH = "original_data.zip"
OUTPUT_PATH = "data/eeg_dataset_raw.csv"

dfs = []

with zipfile.ZipFile(ZIP_PATH, "r") as z:
    for filename in z.namelist():

        # 1. Ignora cartelle e file nascosti di sistema che non sono CSV
        if not filename.endswith(".csv"):
            continue

        parts = filename.replace(".csv", "").split("-")

        # 2. Se il nome non ha 3 parti (soggetto-stato-sessione), saltalo
        if len(parts) < 3:
            print(f"Ignorato file/cartella anomala: {filename}")
            continue

        subject = parts[0]
        state = parts[1]
        session = int(parts[2])

        # Scarta lo stato relaxed
        if state == "relaxed":
            continue

        # Assegna 1 per concentrating, 0 per neutral
        label = 1 if state == "concentrating" else 0

        # Legge i dati
        with z.open(filename) as f:
            df = pd.read_csv(f)

        # Aggiunge le colonne necessarie
        df["subject"] = subject
        df["session"] = session
        df["label"] = label

        dfs.append(df)
        print(f"Elaborato: {filename}")

# Unisce tutti i file in un'unica tabella
merged = pd.concat(dfs, ignore_index=True)

# Crea l'ordine esatto: [Soggetto] + [Sessione] + [TUTTE LE METRICHE] + [Label finale]
metric_cols = [col for col in merged.columns if col not in ["subject", "session", "label"]]
final_cols = ["subject", "session"] + metric_cols + ["label"]

# Applica il nuovo ordine
merged = merged[final_cols]

# Salva il risultato
merged.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinito! Salvato {OUTPUT_PATH} con {len(merged):,} righe totali.")
