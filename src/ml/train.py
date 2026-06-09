"""
Ensemble ML Pipeline — SVM + Random Forest + Isolation Forest
--------------------------------------------------------------
Flusso completo:
  1. KNN Imputation sui NaN/Inf
  2. Grid Search con LOSO per trovare i best_params di ogni modello
  3. LOSO manuale con best_params → 4 .pkl per modello + metriche per fold
  4. Report comparativo finale

Strategia ensemble (prima): tutti e 4 i pkl di ogni modello votano,
poi i voti medi dei modelli vengono combinati in prediction.py.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV

# -- Percorsi ----------------------------------------------------------------
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEATURES_CSV   = os.path.join(BASE_DIR, "data", "final_features.csv")
MODEL_BASE_DIR = os.path.join(BASE_DIR, "src", "ml", "models")

# Scegli quali modelli addestrare
TRAIN_SVM = True
TRAIN_RF  = True
TRAIN_IF  = True   # Isolation Forest (anomaly detection)

# KNN Imputer — numero di vicini per riempire NaN
KNN_NEIGHBORS = 5
# ----------------------------------------------------------------------------


def load_and_impute():
    """
    Carica le feature PSD, sostituisce NaN e Inf con KNN Imputation.

    KNNImputer cerca le K epoche più simili a quella con il valore mancante
    e usa la loro media pesata per riempire il buco — molto più corretto
    di sostituire con 0 che distorce la distribuzione.
    """
    print("Caricamento features...")
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [c for c in df.columns
                    if c not in ["subject", "session", "epoch_id", "label"]]

    X_raw = df[feature_cols].values.astype(float)

    # Sostituisce Inf con NaN prima dell'imputation (KNNImputer gestisce solo NaN)
    X_raw[~np.isfinite(X_raw)] = np.nan

    n_nan = np.isnan(X_raw).sum()
    if n_nan > 0:
        print(f"  {n_nan} valori NaN/Inf trovati → KNN Imputation (k={KNN_NEIGHBORS})...")
        imputer = KNNImputer(n_neighbors=KNN_NEIGHBORS)
        X = imputer.fit_transform(X_raw)
        # Salva l'imputer per usarlo in prediction.py
        imp_path = os.path.join(MODEL_BASE_DIR, "knn_imputer.pkl")
        os.makedirs(MODEL_BASE_DIR, exist_ok=True)
        joblib.dump(imputer, imp_path)
        print(f"  Imputer salvato: {imp_path}")
    else:
        print("  Nessun NaN trovato — imputation non necessaria.")
        X = X_raw

    y      = df["label"].values
    groups = df["subject"].values
    print(f"  Dataset: {X.shape[0]} epoche × {X.shape[1]} feature")
    print(f"  Label  : {int((y==0).sum())} neutral / {int((y==1).sum())} concentrating")
    return X, y, groups


def clean_subject_name(name):
    return str(name).split("/")[-1].split("\\")[-1].replace(".csv", "")


def train_supervised(name, pipeline, param_grid, X, y, groups):
    """
    Grid Search + LOSO per modelli supervisionati (SVM, RF).
    Salva 4 pkl (uno per fold) e un report testuale.
    """
    logo    = LeaveOneGroupOut()
    out_dir = os.path.join(MODEL_BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  MODELLO SUPERVISIONATO: {name.upper()}")
    print(f"{'='*55}")

    # -- 1. Grid Search per trovare i migliori iperparametri ------------------
    # La LOSO interna valuta ogni combinazione su tutti e 4 i fold
    # e sceglie quella con F1 medio più alto
    print("  Grid Search in corso (LOSO interna su tutto il dataset)...")
    grid = GridSearchCV(
        pipeline, param_grid,
        cv=logo,            # LOSO come validazione interna
        scoring="f1",       # F1 perché il dataset è sbilanciato
        n_jobs=-1,          # usa tutti i core disponibili
        verbose=0
    )
    grid.fit(X, y, groups=groups)
    print(f"  Parametri ottimali : {grid.best_params_}")
    print(f"  F1 medio (GS)      : {grid.best_score_*100:.1f}%")

    # -- 2. LOSO manuale con best_params --------------------------------------
    # Qui vengono addestrati e salvati i 4 modelli reali
    print(f"\n  LOSO manuale — addestramento e salvataggio 4 fold...")
    fold_results   = []
    all_y_true     = []
    all_y_pred     = []

    for train_idx, test_idx in logo.split(X, y, groups):
        test_subj = clean_subject_name(groups[test_idx][0])
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        # Clone: copia la pipeline con best_params ma senza pesi → readdestra
        fold_model = clone(grid.best_estimator_)
        fold_model.fit(X_train, y_train)
        y_pred = fold_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        fold_results.append({
            "subject": test_subj, "accuracy": acc, "f1": f1,
            "n_test": len(y_test), "n_concentrating": int(y_test.sum())
        })
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Salva il modello di questo fold
        pkl_path = os.path.join(out_dir, f"model_excluding_{test_subj}.pkl")
        joblib.dump(fold_model, pkl_path)

        print(f"    fold '{test_subj:10s}'  "
              f"acc={acc*100:.1f}%  f1={f1*100:.1f}%  "
              f"({len(y_test)} epoche, {int(y_test.sum())} concentrating)  "
              f"→ {os.path.basename(pkl_path)}")

    # -- 3. Metriche globali aggregate su tutti i fold ------------------------─
    global_acc = accuracy_score(all_y_true, all_y_pred)
    global_f1  = f1_score(all_y_true, all_y_pred, zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred)
    cr = classification_report(all_y_true, all_y_pred,
                                target_names=["neutral", "concentrating"],
                                zero_division=0)

    print(f"\n  Globali {name.upper()}: accuracy={global_acc*100:.1f}%  f1={global_f1*100:.1f}%")
    print(f"  Confusion matrix:\n{cm}")

    _save_report(name, out_dir, grid.best_params_, grid.best_score_,
                 fold_results, global_acc, global_f1, cm, cr)

    return {"name": name, "accuracy": global_acc, "f1": global_f1,
            "best_params": grid.best_params_, "fold_results": fold_results}


def train_isolation_forest(X, y, groups):
    """
    Isolation Forest — anomaly detection.

    Logica: addestra solo sulle epoche NEUTRAL (label=0) per imparare
    la baseline. Le epoche concentrating sono trattate come anomalie.
    Alla predizione restituisce uno anomaly score che viene convertito
    in probabilità per l'ensemble.

    LOSO: per ogni fold, addestra IF solo sui neutral del train set,
    poi valuta su tutto il test set (neutral + concentrating).
    """
    name    = "isolation_forest"
    out_dir = os.path.join(MODEL_BASE_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    logo    = LeaveOneGroupOut()

    print(f"\n{'='*55}")
    print(f"  ANOMALY DETECTION: ISOLATION FOREST")
    print(f"{'='*55}")
    print("  Logica: addestrato solo su neutral → concentrating = anomalia")

    # Grid Search manuale su contamination
    # contamination = proporzione attesa di anomalie nel training set
    # nel nostro caso è 0 (addestriamo solo su neutral) ma influenza il threshold
    contamination_values = [0.05, 0.1, 0.2]
    best_contamination   = 0.1
    best_f1_gs           = -1.0

    print(f"\n  Grid Search su contamination {contamination_values}...")
    for cont in contamination_values:
        fold_f1s = []
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train_all = X[train_idx]
            y_train_all = y[train_idx]
            X_test      = X[test_idx]
            y_test      = y[test_idx]

            # Addestra SOLO sui neutral del train
            X_train_neutral = X_train_all[y_train_all == 0]
            if len(X_train_neutral) == 0:
                continue

            clf = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
            clf.fit(X_train_neutral)

            # predict: +1 = inlier (neutral), -1 = outlier (concentrating)
            # convertiamo: -1 → label 1 (concentrating), +1 → label 0 (neutral)
            raw_pred = clf.predict(X_test)
            y_pred   = np.where(raw_pred == -1, 1, 0)
            fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

        mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else 0.0
        print(f"    contamination={cont:.2f} → F1 medio={mean_f1*100:.1f}%")
        if mean_f1 > best_f1_gs:
            best_f1_gs         = mean_f1
            best_contamination = cont

    print(f"  Contamination ottimale: {best_contamination}  (F1={best_f1_gs*100:.1f}%)")

    # -- LOSO manuale con best contamination ----------------------------------
    print(f"\n  LOSO manuale — salvataggio 4 fold...")
    fold_results = []
    all_y_true   = []
    all_y_pred   = []

    for train_idx, test_idx in logo.split(X, y, groups):
        test_subj       = clean_subject_name(groups[test_idx][0])
        X_train_all     = X[train_idx]
        y_train_all     = y[train_idx]
        X_test          = X[test_idx]
        y_test          = y[test_idx]

        X_train_neutral = X_train_all[y_train_all == 0]

        clf = IsolationForest(contamination=best_contamination,
                              random_state=42, n_jobs=-1)
        clf.fit(X_train_neutral)

        raw_pred = clf.predict(X_test)
        y_pred   = np.where(raw_pred == -1, 1, 0)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, zero_division=0)
        fold_results.append({
            "subject": test_subj, "accuracy": acc, "f1": f1,
            "n_test": len(y_test), "n_concentrating": int(y_test.sum())
        })
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        pkl_path = os.path.join(out_dir, f"model_excluding_{test_subj}.pkl")
        joblib.dump(clf, pkl_path)

        print(f"    fold '{test_subj:10s}'  "
              f"acc={acc*100:.1f}%  f1={f1*100:.1f}%  "
              f"({len(y_test)} epoche, {int(y_test.sum())} concentrating)  "
              f"→ {os.path.basename(pkl_path)}")

    global_acc = accuracy_score(all_y_true, all_y_pred)
    global_f1  = f1_score(all_y_true, all_y_pred, zero_division=0)
    cm = confusion_matrix(all_y_true, all_y_pred)
    cr = classification_report(all_y_true, all_y_pred,
                                target_names=["neutral", "concentrating"],
                                zero_division=0)

    print(f"\n  Globali IF: accuracy={global_acc*100:.1f}%  f1={global_f1*100:.1f}%")
    print(f"  Confusion matrix:\n{cm}")

    # Salva contamination ottimale per prediction.py
    meta_path = os.path.join(out_dir, "meta.pkl")
    joblib.dump({"best_contamination": best_contamination}, meta_path)

    _save_report(name, out_dir,
                 {"contamination": best_contamination}, best_f1_gs,
                 fold_results, global_acc, global_f1, cm, cr)

    return {"name": name, "accuracy": global_acc, "f1": global_f1,
            "best_params": {"contamination": best_contamination},
            "fold_results": fold_results}


def _save_report(name, out_dir, best_params, best_score_gs,
                 fold_results, global_acc, global_f1, cm, cr):
    """Salva il report testuale di un modello."""
    path = os.path.join(out_dir, f"report_{name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"=== LOSO REPORT: {name.upper()} ===\n\n")
        f.write(f"Parametri ottimali  : {best_params}\n")
        f.write(f"F1 medio GridSearch : {best_score_gs*100:.1f}%\n\n")
        f.write("-- Risultati per fold --\n")
        for r in fold_results:
            f.write(f"  test su {r['subject']:12s}  "
                    f"acc={r['accuracy']*100:.1f}%  f1={r['f1']*100:.1f}%  "
                    f"({r['n_test']} epoche, {r['n_concentrating']} concentrating)\n")
        f.write(f"\n-- Globali --\n")
        f.write(f"Accuracy : {global_acc*100:.1f}%\n")
        f.write(f"F1       : {global_f1*100:.1f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm) + "\n\n")
        f.write("Classification Report:\n")
        f.write(cr)


def print_comparative_report(results):
    """Stampa e salva il confronto finale tra tutti i modelli."""
    print(f"\n{'='*55}")
    print("  REPORT COMPARATIVO — ENSEMBLE ETEROGENEO")
    print(f"{'='*55}")
    print(f"  {'Modello':<20} {'Accuracy':>10} {'F1':>10}")
    print(f"  {'-'*42}")
    for r in sorted(results, key=lambda x: x["f1"], reverse=True):
        print(f"  {r['name'].upper():<20} "
              f"{r['accuracy']*100:>9.1f}%  {r['f1']*100:>9.1f}%")

    print(f"\n  Soglia consigliata per inclusione nell'ensemble: F1 > 55%")
    worthy = [r for r in results if r["f1"] > 0.55]
    if worthy:
        print(f"  Modelli sopra soglia: {[r['name'].upper() for r in worthy]}")
    else:
        print("  Nessun modello supera la soglia — rivedi il preprocessing.")

    comp_path = os.path.join(MODEL_BASE_DIR, "comparative_report.txt")
    with open(comp_path, "w", encoding="utf-8") as f:
        f.write("=== REPORT COMPARATIVO ENSEMBLE ETEROGENEO ===\n\n")
        f.write(f"{'Modello':<20} {'Accuracy':>10} {'F1':>10}\n")
        f.write("-" * 42 + "\n")
        for r in sorted(results, key=lambda x: x["f1"], reverse=True):
            f.write(f"{r['name'].upper():<20} "
                    f"{r['accuracy']*100:>9.1f}%  {r['f1']*100:>9.1f}%\n")
        f.write("\nParametri ottimali per modello:\n")
        for r in results:
            f.write(f"  {r['name'].upper()}: {r['best_params']}\n")
        f.write("\nRisultati per fold:\n")
        for r in results:
            f.write(f"\n  {r['name'].upper()}:\n")
            for fold in r["fold_results"]:
                f.write(f"    test su {fold['subject']:12s}  "
                        f"acc={fold['accuracy']*100:.1f}%  "
                        f"f1={fold['f1']*100:.1f}%\n")
    print(f"\n  Report salvato: {comp_path}")


def main():
    print("=" * 55)
    print("  TRAINING PIPELINE — ENSEMBLE ETEROGENEO")
    print("=" * 55)

    X, y, groups = load_and_impute()
    results = []

    if TRAIN_SVM:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(class_weight="balanced", probability=True, random_state=42)),
        ])
        param_grid = {
            "clf__C":      [0.1, 1, 10],
            "clf__kernel": ["rbf", "linear"],
        }
        results.append(train_supervised("svm", pipeline, param_grid, X, y, groups))

    if TRAIN_RF:
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(class_weight="balanced", random_state=42)),
        ])
        param_grid = {
            "clf__n_estimators": [100, 300],
            "clf__max_depth":    [None, 10],
        }
        results.append(train_supervised("rf", pipeline, param_grid, X, y, groups))

    if TRAIN_IF:
        results.append(train_isolation_forest(X, y, groups))

    if results:
        print_comparative_report(results)


if __name__ == "__main__":
    main()
