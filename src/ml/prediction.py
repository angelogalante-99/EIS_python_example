"""
Prediction — da feature a verdetto
-------------------------------------
Modulo "puro": riceve un vettore di 6 feature (già estratte da
features_extraction.extract_features(), chiamata dal main) e
restituisce un verdetto tramite voto di maggioranza tra i 3
model_final.pkl (SVM, LDA, LightGBM).

Non fa I/O (streaming.py), non fa preprocessing (preprocessing.py),
non fa estrazione feature (features_extraction.py) — solo
"feature in -> verdetto out".
"""

import os
import joblib
import numpy as np

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_BASE_DIR = os.path.join(BASE_DIR, "src", "ml", "models")

LABEL_MAP = {0: "neutral", 1: "concentrating"}

# Modelli da caricare — ognuno ha un model_final.pkl addestrato su
# tutti i soggetti (a+b+d, subjectc escluso — vedi train.py)
MODEL_NAMES = ["svm", "lda", "lgbm"]


def load_models():
    """Carica i model_final.pkl per ciascun modello (vedi MODEL_NAMES)."""
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODEL_BASE_DIR, name, "model_final.pkl")
        if not os.path.exists(path):
            print(f"  ⚠ Modello non trovato: {path} — escluso dal voto.")
            continue
        models[name] = joblib.load(path)
        print(f"  Caricato: {name.upper()} ← {path}")
    if not models:
        raise FileNotFoundError(
            "Nessun model_final.pkl trovato. Esegui prima train.py.")
    return models


def predict_epoch(models, features):
    """
    Predice lo stato a partire da un vettore di 6 feature.

    features: array-like di 6 valori, ordine = FEATURE_NAMES
              (vedi features_extraction.py)

    Voto di maggioranza: se >= metà vota concentrating (1) -> concentrating.
    In caso di pareggio esatto decide in base alla probabilità media.

    Ritorna: (verdetto:int, dettagli:dict con votes/probs/n_concentrating/n_models)
    """
    X = np.asarray(features).reshape(1, -1)
    votes, probs = {}, {}

    for name, model in models.items():
        pred = int(model.predict(X)[0])
        votes[name] = pred
        if hasattr(model, "predict_proba"):
            probs[name] = float(model.predict_proba(X)[0][1])  # P(concentrating)

    n_concentrating = sum(votes.values())
    n_models        = len(votes)

    if n_concentrating * 2 > n_models:
        verdict = 1
    elif n_concentrating * 2 < n_models:
        verdict = 0
    else:
        mean_prob = np.mean(list(probs.values())) if probs else 0.5
        verdict = 1 if mean_prob >= 0.5 else 0

    return verdict, {"votes": votes, "probs": probs,
                     "n_concentrating": n_concentrating, "n_models": n_models}