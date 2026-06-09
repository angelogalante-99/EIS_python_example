"""
Inference Pipeline - Subject & State Filtering
----------------------------------------------
Carica i modelli dell'ensemble, seleziona un'epoca reale filtrando per
Soggetto e Stato Cognitivo, e calcola la media probabilistica del verdetto.
"""

import os
import joblib
import numpy as np
import pandas as pd
import glob

# Configurazione percorsi
MODEL_DIR = os.path.join("src", "ml", "models", "rf")
CSV_PATH  = os.path.join("data", "final_features.csv")

def predict_with_ensemble(new_epoch_features):
    """Interroga i 4 modelli della LOSO, media le probabilità e restituisce il verdetto."""
    model_files = glob.glob(os.path.join(MODEL_DIR, "model_excluding_*.pkl"))
    
    if not model_files:
        print("Errore: Nessun modello trovato nella cartella.")
        return
        
    print(f"\nModelli caricati nell'Ensemble: {len(model_files)}")
    all_probabilities = []
    
    for model_path in model_files:
        model = joblib.load(model_path)
        # Estraiamo la probabilità per la classe 1 (Ansia/Concentrazione)
        prob = model.predict_proba([new_epoch_features])[0][1] 
        all_probabilities.append(prob)
        
        model_name = os.path.basename(model_path)
        print(f"  -> {model_name: <32} vota: {prob * 100:.1f}% Ansia")
        
    # Media delle probabilità
    final_prob = np.mean(all_probabilities)
    print("-" * 55)
    print(f"Probabilità Media dell'Ensemble: {final_prob * 100:.1f}%")
    
    if final_prob >= 0.50:
        print("VERDETTO FINALE: L'utente è in stato di CONCENTRAZIONE/ANSIA (Label 1)")
    else:
        print("VERDETTO FINALE: L'utente è RILASSATO/NEUTRAL (Label 0)")


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Errore: Dataset {CSV_PATH} non trovato.")
        exit()

    df = pd.read_csv(CSV_PATH)
    
    # ── IMPOSTA I TUOI CRITERI DI RICERCA QUI ───────────────────
    TARGET_SUBJECT = "subjecta"  # Opzioni: "subjecta", "subjectb", "subjectc", "subjectd"
    TARGET_LABEL   = 1           # Opzioni: 0 (Rilassato), 1 (Concentrato)
    # ──────────────────────────────────────────────────────────

    # Filtro intelligente: gestisce sia "subjecta" che "original_data/subjecta"
    filtered_df = df[
        (df["subject"].str.contains(TARGET_SUBJECT, case=False)) & 
        (df["label"] == TARGET_LABEL)
    ]

    if filtered_df.empty:
        print(f"Nessuna epoca trovata per il Soggetto '{TARGET_SUBJECT}' con Label {TARGET_LABEL}.")
        print("Controlla i nomi presenti nel tuo CSV.")
        exit()

    # Estraiamo un'epoca casuale tra quelle che soddisfano il filtro
    chosen_epoch = filtered_df.sample(n=1, random_state=42) # random_state fisso per riproducibilità, rimuovilo se vuoi vera casualità
    
    epoch_id = chosen_epoch["epoch_id"].iloc[0]
    subject_real_name = chosen_epoch["subject"].iloc[0]
    vera_label = chosen_epoch["label"].iloc[0]
    
    # Isola le 33 colonne matematiche
    feature_cols = [c for c in df.columns if c not in ["subject", "session", "epoch_id", "label"]]
    veri_dati_eeg = chosen_epoch[feature_cols].values[0]

    print("=" * 60)
    print(f"RICERCA COMPLETATA - ESTRATTA EPOCA CASUALE")
    print(f"  Soggetto Reale : {subject_real_name}")
    print(f"  ID Epoca       : #{epoch_id}")
    print(f"  Stato Reale    : {'CONCENTRATO/ANSIA (1)' if vera_label == 1 else 'RILASSATO/NEUTRAL (0)'}")
    print("=" * 60)

    # Avvia la predizione
    predict_with_ensemble(veri_dati_eeg)