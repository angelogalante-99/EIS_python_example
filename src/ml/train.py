"""
Ensemble ML Pipeline: Grid Search + LOSO Ensemble Saving
--------------------------------------------------------
Addestra la pipeline, trova i migliori iperparametri, e salva un modello 
per OGNI piega della LOSO (Ensemble Averaging) pronto per la media probabilistica.
"""

import pandas as pd
import numpy as np
import time
import os
import joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV

# --- Scegli il modello qui! ---
MODEL_CHOICE = 'svm'  # Opzioni: 'rf', 'svm', 'minirocket'

FEATURES_CSV  = "data/final_features.csv"
RAW_EEG_CSV   = "data/eeg_dataset_preprocessed.csv"
MODEL_OUT_DIR = os.path.join("src", "ml", "models", MODEL_CHOICE) # Crea una sottocartella per il modello scelto

def load_data_for_model(model_type):
    if model_type in ['rf', 'svm']:
        df = pd.read_csv(FEATURES_CSV)
        feature_cols = [c for c in df.columns if c not in ["subject", "session", "epoch_id", "label"]]
        X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        return X, df["label"].values, df["subject"].values

    elif model_type == 'minirocket':
        df = pd.read_csv(RAW_EEG_CSV)
        df["global_epoch"] = df["subject"] + "_" + df["session"].astype(str) + "_" + df["epoch_id"].astype(str)
        X_temp, y_temp, groups_temp = [], [], []
        
        for _, group in df.groupby("global_epoch"):
            X_temp.append(group[["TP9", "AF7", "AF8", "TP10"]].values.T)
            y_temp.append(group["label"].iloc[0])
            groups_temp.append(group["subject"].iloc[0])
            
        target_len = int(np.bincount([x.shape[1] for x in X_temp]).argmax())
        
        X_list, y_list, groups_list = [], [], []
        for i in range(len(X_temp)):
            if X_temp[i].shape[1] == target_len:
                X_list.append(X_temp[i])
                y_list.append(y_temp[i])
                groups_list.append(groups_temp[i])
                
        return np.array(X_list), np.array(y_list), np.array(groups_list)

def main():
    print(f"=== Avvio Pipeline ENSEMBLE per: {MODEL_CHOICE.upper()} ===")
    X, y, groups = load_data_for_model(MODEL_CHOICE)

    # 1. Pipeline e Parametri
    if MODEL_CHOICE == 'rf':
        pipeline = Pipeline([
            ('scaler', StandardScaler()), 
            ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
        ])
        param_grid = {'clf__n_estimators': [100, 300], 'clf__max_depth': [None, 10]}
        
    elif MODEL_CHOICE == 'svm':
        # probability=True è FONDAMENTALE per l'ensemble
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(class_weight='balanced', probability=True, random_state=42))
        ])
        param_grid = {'clf__C': [0.1, 1, 10], 'clf__kernel': ['rbf', 'linear']}
        
    elif MODEL_CHOICE == 'minirocket':
        from sktime.transformations.panel.rocket import MiniRocketMultivariate
        # Usiamo LogisticRegression invece di Ridge per avere predict_proba()
        pipeline = Pipeline([
            ('rocket', MiniRocketMultivariate()),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced', max_iter=2000))
        ])
        param_grid = {'clf__C': [0.1, 1.0, 10.0]}

    # 2. Grid Search per trovare i migliori parametri
    logo = LeaveOneGroupOut()
    grid = GridSearchCV(pipeline, param_grid, cv=logo, scoring='accuracy', n_jobs=-1)
    
    print("\nRicerca dei migliori iperparametri in corso...")
    grid.fit(X, y, groups=groups)
    print(f"Parametri vincenti: {grid.best_params_}")

    # 3. SALVATAGGIO DELL'ENSEMBLE
    print("\n" + "="*50)
    print("CREAZIONE DEI MODELLI ENSEMBLE (LOSO)...")
    os.makedirs(MODEL_OUT_DIR, exist_ok=True)
    
    unique_groups = np.sort(np.unique(groups))
    
    for train_idx, test_idx in logo.split(X, y, groups):
        # Chi è il soggetto lasciato fuori da questo specifico modello?
        test_subj = groups[test_idx][0]
        
        # --- FIX: Pulizia del nome ---
        # Rimuove cartelle dal nome (es. "original_data/subjecta" diventa "subjecta")
        safe_subj_name = str(test_subj).split('/')[-1].split('\\')[-1].replace('.csv', '')
        
        X_train, y_train = X[train_idx], y[train_idx]
        
        # Cloniamo la pipeline vincente (fresca, senza addestramento)
        fold_model = clone(grid.best_estimator_)
        
        # Addestriamo il modello su N-1 soggetti
        fold_model.fit(X_train, y_train)
        
        # Salviamo questo specifico modello
        model_filename = os.path.join(MODEL_OUT_DIR, f"model_excluding_{safe_subj_name}.pkl")
        joblib.dump(fold_model, model_filename)
        print(f"  [OK] Modello addestrato senza '{safe_subj_name}' salvato in: {model_filename}")

    # Report finale
    report_filename = os.path.join(MODEL_OUT_DIR, f"report_{MODEL_CHOICE}.txt")
    with open(report_filename, "w") as f:
        f.write(f"=== ENSEMBLE REPORT: {MODEL_CHOICE.upper()} ===\n")
        f.write(f"Iperparametri ottimali: {grid.best_params_}\n")
        f.write(f"Accuratezza Media Stimata (LOSO): {grid.best_score_ * 100:.1f}%\n")
    print(f"\nReport completato: {report_filename}")

if __name__ == "__main__":
    main()