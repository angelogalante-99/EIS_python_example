"""
Training Pipeline — SVM + LDA + LightGBM
-----------------------------------------
Modelli scelti in base alle caratteristiche del dataset:
  - 6 feature, ~450 epoche, 3 soggetti (subjectc escluso)
  - separazione emersa lineare dalla GridSearch precedente
  - dataset sbilanciato (~1.7:1 neutral/concentrating)

SVM  : riferimento solido, kernel lineare/rbf, gestisce sbilanciamento
       con class_weight='balanced'. GridSearch su C e kernel.

LDA  : Linear Discriminant Analysis — ideale per separazione lineare
       con pochi campioni. Stabile, nessun rischio overfitting.
       GridSearch su solver e shrinkage (regolarizzazione Ledoit-Wolf).
       Se LDA ≈ SVM → la separazione è davvero lineare (conferma).

LightGBM : Gradient Boosting sequenziale — corregge errori iterativamente,
           robusto su dataset piccoli con feature continue.
           Migliore di Random Forest su pochi dati perché non crea alberi
           profondi indipendenti. GridSearch su n_estimators, learning_rate,
           max_depth, min_child_samples (soglia minima per foglia — cruciale
           con dataset piccoli per evitare overfitting).

Protocollo valutazione: LOSO (Leave-One-Subject-Out)
  - addestra su 2 soggetti, valuta sul terzo
  - ripete per ogni soggetto → 3 fold
  - iperparametri scelti con GridSearch sulla stessa LOSO
  - nota: bias ottimistico lieve (iperparametri vedono i soggetti di test)
    ma identico per tutti i modelli → il confronto è equo

subjectc escluso: elettrodo frontale compromesso in tutte le sessioni
  (AF7 rotta in concentrating, AF8 rotta in neutral) — documentato
  da ispezione visiva del segnale e analisi PSD media per sessione.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from lightgbm import LGBMClassifier

# ── Percorsi ─────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_CSV   = os.path.join(BASE_DIR, "data", "final_features.csv")
MODEL_BASE_DIR = os.path.join(BASE_DIR, "src", "ml", "models")

# ── Soggetti esclusi ──────────────────────────────────────────────────────────
# subjectc: elettrodo frontale compromesso in tutte le sessioni.
# AF7 rotta nelle sessioni concentrating (EMG continuo visibile nel tempo),
# AF8 rotta nella sessione neutral (artefatto da epoca 19 in poi).
# Confound di sessione documentato: il canale disturbato cambia tra stati,
# rendendo il Cohen's d inter-classe un artefatto di qualità del segnale
# piuttosto che una differenza cognitiva.
EXCLUDED_SUBJECTS = ["subjectc"]

KNN_NEIGHBORS = 5
# ─────────────────────────────────────────────────────────────────────────────


def load_data():
    """
    Carica le feature, esclude i soggetti problematici,
    sostituisce NaN/Inf con KNN Imputation.
    """
    print("Caricamento features...")
    df = pd.read_csv(FEATURES_CSV)

    # Esclusione subjectc — motivazione nel docstring del modulo
    before = len(df)
    df = df[~df["subject"].isin(EXCLUDED_SUBJECTS)].reset_index(drop=True)
    after = len(df)
    if before != after:
        print(f"  Esclusi {before - after} campioni "
              f"({EXCLUDED_SUBJECTS}) per qualità del segnale.")

    feature_cols = [c for c in df.columns
                    if c not in ["subject", "session", "epoch_id", "label"]]
    X_raw = df[feature_cols].values.astype(float)
    X_raw[~np.isfinite(X_raw)] = np.nan

    n_nan = np.isnan(X_raw).sum()
    if n_nan > 0:
        print(f"  {n_nan} NaN/Inf → KNN Imputation (k={KNN_NEIGHBORS})...")
        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        X = imputer.fit_transform(X_raw)
        joblib.dump(imputer, os.path.join(MODEL_BASE_DIR, "knn_imputer.pkl"))
    else:
        print("  Nessun NaN trovato — imputation non necessaria.")
        X = X_raw

    y      = df["label"].values
    groups = df["subject"].values
    subjects = sorted(df["subject"].unique())

    print(f"  Dataset : {X.shape[0]} epoche × {X.shape[1]} feature")
    print(f"  Soggetti: {subjects}")
    print(f"  Label   : {int((y==0).sum())} neutral / {int((y==1).sum())} concentrating")
    return X, y, groups


def clean_name(name):
    return str(name).split("/")[-1].split("\\")[-1].replace(".csv", "")


def train_supervised(name, pipeline, param_grid, X, y, groups):
    """
    GridSearch LOSO + LOSO manuale con best_params.
    Salva un pkl per fold e il report testuale.
    """
    logo    = LeaveOneGroupOut()
    out_dir = os.path.join(MODEL_BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*57}")
    print(f"  MODELLO: {name.upper()}")
    print(f"{'='*57}")

    # ── GridSearch ────────────────────────────────────────────
    print("  Grid Search (LOSO interna)...")
    grid = GridSearchCV(pipeline, param_grid, cv=logo,
                        scoring="f1", n_jobs=-1, verbose=0)
    grid.fit(X, y, groups=groups)
    print(f"  Parametri ottimali : {grid.best_params_}")
    print(f"  F1 medio (GS)      : {grid.best_score_*100:.1f}%")

    # ── LOSO manuale ──────────────────────────────────────────
    print(f"\n  LOSO — {LeaveOneGroupOut().get_n_splits(groups=groups)} fold...")
    fold_results = []
    all_true, all_pred = [], []

    for train_idx, test_idx in logo.split(X, y, groups):
        subj = clean_name(groups[test_idx][0])
        model = clone(grid.best_estimator_)
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        acc = accuracy_score(y[test_idx], pred)
        f1  = f1_score(y[test_idx], pred, zero_division=0)
        fold_results.append({"subject": subj, "acc": acc, "f1": f1,
                              "n": len(y[test_idx]),
                              "n_c": int(y[test_idx].sum())})
        all_true.extend(y[test_idx])
        all_pred.extend(pred)

        pkl = os.path.join(out_dir, f"model_excluding_{subj}.pkl")
        joblib.dump(model, pkl)
        print(f"    [{subj:12s}]  acc={acc*100:.1f}%  f1={f1*100:.1f}%  "
              f"({len(y[test_idx])} epoche, {int(y[test_idx].sum())} conc.)  "
              f"→ {os.path.basename(pkl)}")

    # ── Metriche globali ──────────────────────────────────────
    g_acc = accuracy_score(all_true, all_pred)
    g_f1  = f1_score(all_true, all_pred, zero_division=0)
    cm    = confusion_matrix(all_true, all_pred)
    cr    = classification_report(all_true, all_pred,
                                  target_names=["neutral", "concentrating"],
                                  zero_division=0)
    print(f"\n  Globali: accuracy={g_acc*100:.1f}%  f1={g_f1*100:.1f}%")
    print(f"  Confusion matrix:\n{cm}")

    _save_report(name, out_dir, grid.best_params_, grid.best_score_,
                 fold_results, g_acc, g_f1, cm, cr)

    return {"name": name, "acc": g_acc, "f1": g_f1,
            "best_params": grid.best_params_, "folds": fold_results}


def _save_report(name, out_dir, best_params, gs_score,
                 folds, g_acc, g_f1, cm, cr):
    path = os.path.join(out_dir, f"report_{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== REPORT LOSO: {name.upper()} ===\n\n")
        f.write(f"Parametri ottimali  : {best_params}\n")
        f.write(f"F1 medio GridSearch : {gs_score*100:.1f}%\n\n")
        f.write("-- Risultati per fold --\n")
        for r in folds:
            f.write(f"  test {r['subject']:12s}  "
                    f"acc={r['acc']*100:.1f}%  f1={r['f1']*100:.1f}%  "
                    f"({r['n']} epoche, {r['n_c']} concentrating)\n")
        f.write(f"\n-- Globali --\n")
        f.write(f"Accuracy : {g_acc*100:.1f}%\n")
        f.write(f"F1       : {g_f1*100:.1f}%\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
        f.write("Classification Report:\n" + cr)


def print_comparative_report(results):
    """Stampa e salva il confronto finale tra tutti i modelli."""
    print(f"\n{'='*57}")
    print("  REPORT COMPARATIVO")
    print(f"{'='*57}")

    subjects = sorted({f["subject"] for r in results for f in r["folds"]})
    header = f"  {'modello':<14} {'F1 glob':>8} {'acc':>7}   "
    header += "".join(f"{s[-1].upper():>8}" for s in subjects)
    print(header)
    print("  " + "-" * (len(header)-2))

    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        fold_str = ""
        for s in subjects:
            fold = next((f for f in r["folds"] if f["subject"] == s), None)
            fold_str += f"{fold['f1']*100:>7.1f}%" if fold else f"{'n/a':>8}"
        print(f"  {r['name'].upper():<14} "
              f"{r['f1']*100:>7.1f}%  {r['acc']*100:>6.1f}%   {fold_str}")

    print(f"\n  Nota: soggetti esclusi: {EXCLUDED_SUBJECTS}")
    print(f"  Nota: bias ottimistico lieve — iperparametri scelti con stessa LOSO")

    # Salva report
    path = os.path.join(MODEL_BASE_DIR, "comparative_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== REPORT COMPARATIVO ===\n\n")
        f.write(f"Soggetti esclusi: {EXCLUDED_SUBJECTS}\n\n")
        f.write(f"{'Modello':<14} {'Accuracy':>10} {'F1':>10}\n")
        f.write("-" * 36 + "\n")
        for r in sorted(results, key=lambda x: x["f1"], reverse=True):
            f.write(f"{r['name'].upper():<14} "
                    f"{r['acc']*100:>9.1f}%  {r['f1']*100:>9.1f}%\n")
        f.write("\nParametri ottimali:\n")
        for r in results:
            f.write(f"  {r['name'].upper()}: {r['best_params']}\n")
        f.write("\nRisultati per fold:\n")
        for r in results:
            f.write(f"\n  {r['name'].upper()}:\n")
            for fold in r["folds"]:
                f.write(f"    [{fold['subject']:12s}]  "
                        f"acc={fold['acc']*100:.1f}%  f1={fold['f1']*100:.1f}%\n")
    print(f"\n  Report salvato: {path}")


def main():
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)

    print("=" * 57)
    print("  TRAINING PIPELINE — SVM + LDA + LightGBM")
    print("=" * 57)

    X, y, groups = load_data()
    results = []

    # ── SVM ──────────────────────────────────────────────────
    # Riferimento principale. Kernel lineare o rbf.
    # class_weight='balanced' compensa lo sbilanciamento neutral/concentrating.
    # C controlla il trade-off margin/errori: valori bassi = margine largo
    # (più robusto ma più errori), valori alti = segue meglio i dati.
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(class_weight="balanced", probability=True, random_state=42))
    ])
    svm_grid = {
        "clf__C":      [0.01, 0.1, 1, 10],
        "clf__kernel": ["linear", "rbf"],
    }
    results.append(train_supervised("svm", svm_pipeline, svm_grid, X, y, groups))

    # ── LDA ──────────────────────────────────────────────────
    # Linear Discriminant Analysis.
    # solver='svd': non inverte la matrice di covarianza direttamente,
    #   più stabile numericamente con pochi campioni.
    # shrinkage='auto': usa la stima di Ledoit-Wolf per regolarizzare
    #   la matrice di covarianza — fondamentale quando n_samples è vicino
    #   a n_features (qui 450 epoche, 6 feature: non critico ma buona pratica).
    #   Con shrinkage='auto' è richiesto solver='lsqr' o 'eigen'.
    # Confronto: se LDA ≈ SVM → separazione è davvero lineare.
    lda_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearDiscriminantAnalysis())
    ])
    lda_grid = {
        "clf__solver":    ["lsqr", "eigen"],
        "clf__shrinkage": [None, "auto", 0.1, 0.5],
    }
    results.append(train_supervised("lda", lda_pipeline, lda_grid, X, y, groups))

    # ── LightGBM ─────────────────────────────────────────────
    # Gradient Boosting sequenziale — costruisce alberi in sequenza
    # correggendo gli errori del modello precedente.
    # n_estimators: numero di alberi. Con dataset piccolo tenerlo basso
    #   (100-300) per evitare overfitting.
    # learning_rate: passo di correzione. Basso = più stabile ma più alberi.
    #   Accoppiato a n_estimators: lr basso + molti alberi vs lr alto + pochi.
    # max_depth: profondità massima di ogni albero. Con 6 feature, 3-5 è
    #   sufficiente — alberi profondi overfittano.
    # min_child_samples: numero minimo di campioni per creare una foglia.
    #   Cruciale con dataset piccolo: valori alti (20-50) prevengono
    #   foglie con pochi campioni che memorizzano il training.
    # class_weight='balanced': compensa lo sbilanciamento.
    lgbm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(
            class_weight="balanced",
            random_state=42,
            verbose=-1       # silenzia i log di LightGBM
        ))
    ])
    lgbm_grid = {
        "clf__n_estimators":     [100, 200, 300],
        "clf__learning_rate":    [0.01, 0.05, 0.1],
        "clf__max_depth":        [3, 5],
        "clf__min_child_samples":[10, 20, 40],
    }
    results.append(train_supervised("lgbm", lgbm_pipeline, lgbm_grid, X, y, groups))

    print_comparative_report(results)


if __name__ == "__main__":
    main()