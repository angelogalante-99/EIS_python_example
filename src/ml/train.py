import sys
import os

# FONDAMENTALE: Dice a Python di considerare la cartella principale del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import re
import pickle
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import joblib

# Importazioni per MNE e Preprocessing
import mne
from src.preprocessing.preprocessing import Preprocessing

# Importazione per MiniRocket
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# Percorsi base
DATA_PKL   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataframe.pkl')
LABELS_XLS = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Video_selezionati.xlsx')

# --- NUOVO: Percorsi separati per ciascun modello ---
SVM_MODEL_PATH        = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'svm_model.pkl')
MINIROCKET_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'minirocket_model.pkl')

# Costanti EEG
AF7_IDX = 0
AF8_IDX = 3
SFREQ = 256
THETA = (4, 8)
ALPHA = (8, 13)
BETA  = (13, 30)

REJECTION_THRESHOLD = 500.0 

CORRUPTED = {('sub-ID021', 'ses-S023')}
LOW_SFREQ = {('sub-ID015', 'ses-S033'): 226, ('sub-ID015', 'ses-S038'): 217}

def _build_label_map(xls_path: str) -> dict[int, int]:
    df = pd.read_excel(xls_path, header=0)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)
    label_map = {}
    for _, row in df.iterrows():
        try:
            exp_id = int(row['Experiment_id'])
            vaq = int(row['VAQ_Estimate'])
            label_map[exp_id] = 0 if vaq <= 2 else 1
        except (ValueError, TypeError):
            continue
    return label_map

def _session_to_exp_id(session_id: str) -> int:
    clean_id = session_id.split('\\')[-1].split('/')[-1]
    match = re.search(r'\d+', clean_id)
    if match:
        return int(match.group())
    return 0

def _band_power(signal: np.ndarray, low: float, high: float, fs: int) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(fs * 2, len(signal)))
    idx = np.logical_and(freqs >= low, freqs <= high)
    return float(np.trapz(psd[idx], freqs[idx]))

def _extract_features(time_series: np.ndarray, fs: int = SFREQ) -> np.ndarray:
    af7 = time_series[AF7_IDX]
    af8 = time_series[AF8_IDX]
    af7_theta = _band_power(af7, *THETA, fs)
    af7_alpha = _band_power(af7, *ALPHA, fs)
    af7_beta  = _band_power(af7, *BETA,  fs)
    af8_alpha = _band_power(af8, *ALPHA, fs)
    tbr = af7_theta / (af7_beta + 1e-10)
    faa = np.log(af8_alpha + 1e-10) - np.log(af7_alpha + 1e-10)
    return np.array([tbr, faa])

def _apply_mne_preprocessing(ts: np.ndarray, fs: float) -> np.ndarray:
    preprocessor = Preprocessing()
    mne.set_log_level('WARNING')
    info = mne.create_info(ch_names=['TP9', 'AF7', 'AF8', 'TP10'], sfreq=fs, ch_types=['eeg'] * 4)
    raw_mne = mne.io.RawArray(ts, info, verbose=False)
    preprocessor.notch_filter(raw_mne, notch_frequency=50.0)
    preprocessor.band_pass_filter(preprocessor.raw, low=1.0, high=40.0)
    return preprocessor.raw.get_data()

def load_dataset_svm():
    with open(DATA_PKL, 'rb') as f:
        data = pickle.load(f)
    label_map = _build_label_map(LABELS_XLS)
    X, y, subjects = [], [], []
    for subj_idx, subj_name in enumerate(sorted(data.keys())):
        for session in data[subj_name]:
            sid = session['session_id']
            if (subj_name, sid) in CORRUPTED: continue
            exp_id = _session_to_exp_id(sid)
            if exp_id not in label_map: continue
            ts = session['time_series']
            fs = LOW_SFREQ.get((subj_name, sid), SFREQ)
            if ts.shape[1] < 10 * fs: continue
            ts_filtered = _apply_mne_preprocessing(ts, fs)
            step = int(4 * fs)
            overlap_step = int(2 * fs)
            for i in range(0, ts_filtered.shape[1] - step, overlap_step):
                epoch_data = ts_filtered[:, i:i+step]
                if np.max(np.abs(epoch_data[AF7_IDX])) > REJECTION_THRESHOLD or \
                   np.max(np.abs(epoch_data[AF8_IDX])) > REJECTION_THRESHOLD:
                    continue 
                feat = _extract_features(epoch_data, fs=fs)
                if not np.isnan(feat).any():
                    X.append(feat)
                    y.append(label_map[exp_id])
                    subjects.append(subj_idx)
    return np.array(X), np.array(y), np.array(subjects)

def load_dataset_minirocket():
    with open(DATA_PKL, 'rb') as f:
        data = pickle.load(f)
    label_map = _build_label_map(LABELS_XLS)
    X, y, subjects = [], [], []
    for subj_idx, subj_name in enumerate(sorted(data.keys())):
        for session in data[subj_name]:
            sid = session['session_id']
            if (subj_name, sid) in CORRUPTED: continue
            exp_id = _session_to_exp_id(sid)
            if exp_id not in label_map: continue
            ts = session['time_series']
            fs = LOW_SFREQ.get((subj_name, sid), SFREQ)
            if ts.shape[1] < 10 * fs: continue
            ts_filtered = _apply_mne_preprocessing(ts, fs)
            step = int(4 * fs)
            overlap_step = int(2 * fs)
            for i in range(0, ts_filtered.shape[1] - step, overlap_step):
                epoch_data = ts_filtered[:, i:i+step]
                raw_af7 = epoch_data[AF7_IDX]
                raw_af8 = epoch_data[AF8_IDX]
                if np.max(np.abs(raw_af7)) > REJECTION_THRESHOLD or \
                   np.max(np.abs(raw_af8)) > REJECTION_THRESHOLD:
                    continue 
                if not np.isnan(raw_af7).any() and not np.isnan(raw_af8).any():
                    X.append(np.vstack((raw_af7, raw_af8)))
                    y.append(label_map[exp_id])
                    subjects.append(subj_idx)
    return np.array(X), np.array(y), np.array(subjects)

def train_svm():
    print("[SVM] Caricamento e Preprocessing dei dati...")
    X, y, _ = load_dataset_svm()
    print(f"[SVM] Dataset pronto: {X.shape[0]} epoche valide.")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    param_grid = {'C': [1, 10], 'gamma': ['scale'], 'kernel': ['rbf']}
    print("[SVM] Avvio Grid Search Rapida...")
    grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, cv=3, verbose=1, n_jobs=-1)
    grid.fit(X_scaled, y)
    
    print(f"[SVM] Migliori parametri: {grid.best_params_}")
    print(f"[SVM] Miglior accuratezza in Validation: {grid.best_score_ * 100:.2f}%")
    
    # Salviamo specificamente nel file della SVM
    output = {'model_type': 'svm', 'model': grid.best_estimator_, 'scaler': scaler}
    joblib.dump(output, SVM_MODEL_PATH)
    print(f"[SVM] Salvato separatamente in: {SVM_MODEL_PATH}")

def train_minirocket():
    print("[MINIROCKET] Caricamento e Preprocessing dei dati...")
    X, y, _ = load_dataset_minirocket()
    print(f"[MINIROCKET] Dataset pronto: {X.shape[0]} epoche valide.")
    
    print("[MINIROCKET] Inizializzazione e Trasformazione (può richiedere tempo)...")
    minirocket = MiniRocketMultivariate()
    X_transformed = minirocket.fit_transform(X)
    
    print("[MINIROCKET] Addestramento Classificatore RidgeCV...")
    classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier.fit(X_transformed, y)
    
    preds = classifier.predict(X_transformed)
    print(f"[MINIROCKET] Accuratezza Training: {accuracy_score(y, preds)*100:.2f}%")
    
    # Salviamo specificamente nel file di MiniRocket
    output = {'model_type': 'minirocket', 'transformer': minirocket, 'classifier': classifier}
    joblib.dump(output, MINIROCKET_MODEL_PATH)
    print(f"[MINIROCKET] Salvato separatamente in: {MINIROCKET_MODEL_PATH}")

if __name__ == '__main__':
    # ==========================================
    # IMPOSTA SUL MODELLO CHE VUOI ADDESTRARE ORA:
    # MODEL_TYPE = 'svm'  oppure  MODEL_TYPE = 'minirocket'
    # ==========================================
    MODEL_TYPE = 'minirocket' 
    
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', 'models'), exist_ok=True)
    if MODEL_TYPE == 'svm':
        train_svm()
    elif MODEL_TYPE == 'minirocket':
        train_minirocket()