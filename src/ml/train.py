import sys
import os

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
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut
import joblib
import mne

from src.preprocessing.preprocessing import Preprocessing
from sktime.transformations.panel.rocket import MiniRocketMultivariate

# --- IMPORT PER EEGNET (PYTORCH) ---
import torch
from skorch import NeuralNetBinaryClassifier
from src.ml.model import EEGNet # Assicurati che model.py contenga la classe EEGNet in PyTorch

# --- PERCORSI ---
DATA_PKL   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataframe.pkl')
LABELS_XLS = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Video_selezionati.xlsx')
SVM_MODEL_PATH        = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'svm_model.pkl')
MINIROCKET_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'minirocket_model.pkl')
EEGNET_MODEL_PATH     = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'eegnet_model.pt')

# --- COSTANTI EEG ---
AF7_IDX, TP9_IDX, TP10_IDX, AF8_IDX = 0, 1, 2, 3
CH_NAMES = ['AF7', 'TP9', 'TP10', 'AF8']

SFREQ = 256
REJECTION_THRESHOLD = 500.0 

THETA, ALPHA, BETA  = (4, 8), (8, 13), (13, 30)

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
    if match: return int(match.group())
    return 0

def _band_power(signal: np.ndarray, low: float, high: float, fs: int) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(fs * 2, len(signal)))
    idx = np.logical_and(freqs >= low, freqs <= high)
    return float(np.trapz(psd[idx], freqs[idx]))

# ==========================================
# PREPROCESSING E FEATURE ENGINEERING (SVM)
# ==========================================
def _apply_mne_preprocessing(ts: np.ndarray, fs: float) -> np.ndarray:
    preprocessor = Preprocessing()
    mne.set_log_level('WARNING')
    info = mne.create_info(ch_names=CH_NAMES, sfreq=fs, ch_types=['eeg'] * 4)
    raw_mne = mne.io.RawArray(ts, info, verbose=False)
    preprocessor.notch_filter(raw_mne, notch_frequency=50.0)
    preprocessor.band_pass_filter(preprocessor.raw, low=1.0, high=45.0)
    return preprocessor.raw.get_data()

def _extract_features(time_series: np.ndarray, fs: int = SFREQ) -> np.ndarray:
    af7, tp9, tp10, af8 = time_series[AF7_IDX], time_series[TP9_IDX], time_series[TP10_IDX], time_series[AF8_IDX]
    tbr_af7 = _band_power(af7, *THETA, fs) / (_band_power(af7, *BETA, fs) + 1e-10)
    tbr_tp9 = _band_power(tp9, *THETA, fs) / (_band_power(tp9, *BETA, fs) + 1e-10)
    tbr_tp10 = _band_power(tp10, *THETA, fs) / (_band_power(tp10, *BETA, fs) + 1e-10)
    tbr_af8 = _band_power(af8, *THETA, fs) / (_band_power(af8, *BETA, fs) + 1e-10)
    faa = np.log(_band_power(af8, *ALPHA, fs) + 1e-10) - np.log(_band_power(af7, *ALPHA, fs) + 1e-10)
    taa = np.log(_band_power(tp10, *ALPHA, fs) + 1e-10) - np.log(_band_power(tp9, *ALPHA, fs) + 1e-10)
    return np.array([tbr_af7, tbr_tp9, tbr_tp10, tbr_af8, faa, taa])

# ==========================================
# DATA LOADERS
# ==========================================
def load_dataset_svm():
    with open(DATA_PKL, 'rb') as f: data = pickle.load(f)
    label_map = _build_label_map(LABELS_XLS)
    X, y, subjects = [], [], []
    for subj_idx, subj_name in enumerate(sorted(data.keys())):
        for session in data[subj_name]:
            sid = session['session_id']
            if ('sub-ID021', 'ses-S023') == (subj_name, sid): continue
            exp_id = _session_to_exp_id(sid)
            if exp_id not in label_map: continue
            ts = session['time_series']
            fs = 226 if sid == 'ses-S033' and subj_name == 'sub-ID015' else (217 if sid == 'ses-S038' and subj_name == 'sub-ID015' else SFREQ)
            if ts.shape[1] < 10 * fs: continue
            ts_filtered = _apply_mne_preprocessing(ts, fs)
            step, overlap_step = int(4 * fs), int(2 * fs)
            for i in range(0, ts_filtered.shape[1] - step, overlap_step):
                epoch_data = ts_filtered[:, i:i+step]
                if np.max(np.abs(epoch_data)) > REJECTION_THRESHOLD: continue 
                feat = _extract_features(epoch_data, fs=fs)
                if not np.isnan(feat).any():
                    X.append(feat); y.append(label_map[exp_id]); subjects.append(subj_idx)
    return np.array(X), np.array(y), np.array(subjects)

def load_dataset_minirocket():
    with open(DATA_PKL, 'rb') as f: data = pickle.load(f)
    label_map = _build_label_map(LABELS_XLS)
    X, y, subjects = [], [], []
    for subj_idx, subj_name in enumerate(sorted(data.keys())):
        for session in data[subj_name]:
            sid = session['session_id']
            if ('sub-ID021', 'ses-S023') == (subj_name, sid): continue
            exp_id = _session_to_exp_id(sid)
            if exp_id not in label_map: continue
            ts = session['time_series']
            fs = 226 if sid == 'ses-S033' and subj_name == 'sub-ID015' else (217 if sid == 'ses-S038' and subj_name == 'sub-ID015' else SFREQ)
            if ts.shape[1] < 10 * fs: continue
            ts_filtered = _apply_mne_preprocessing(ts, fs)
            step, overlap_step = int(4 * fs), int(2 * fs)
            for i in range(0, ts_filtered.shape[1] - step, overlap_step):
                epoch_data = ts_filtered[:, i:i+step]
                if np.max(np.abs(epoch_data)) > REJECTION_THRESHOLD: continue 
                if not np.isnan(epoch_data).any():
                    X.append(epoch_data); y.append(label_map[exp_id]); subjects.append(subj_idx)
    return np.array(X), np.array(y), np.array(subjects)

def load_dataset_eegnet():
    X, y, subjects = load_dataset_minirocket()
    # PyTorch richiede (Epoche, CanaliImg, CanaliSensore, TimeSteps)
    X = X[:, np.newaxis, :, :] 
    return X.astype(np.float32), y.astype(np.float32), subjects

# ==========================================
# TRAINING PIPELINES
# ==========================================
def train_svm():
    print("="*60)
    print("[SVM] Caricamento ed Estrazione Features Coerente (6 metriche a 4 canali)...")
    X, y, subjects = load_dataset_svm()
    print(f"[SVM] Dataset pronto: {X.shape[0]} epoche valide da {len(np.unique(subjects))} soggetti.")
    
    logo = LeaveOneGroupOut()
    accuracies = []
    param_grid = {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto'], 'kernel': ['rbf']}
    
    for train_index, test_index in logo.split(X, y, groups=subjects):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        scaler = StandardScaler()
        X_train_scaled, X_test_scaled = scaler.fit_transform(X_train), scaler.transform(X_test)
        
        grid = GridSearchCV(SVC(probability=True), param_grid, cv=3, n_jobs=-1, verbose=0)
        grid.fit(X_train_scaled, y_train)
        preds = grid.predict(X_test_scaled)
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"  -> Soggetto {subjects[test_index[0]]:2d} escluso | Acc: {acc*100:5.2f}%")
    
    print(f"\n[SVM] Accuratezza Media Reale (Cross-Subject): {np.mean(accuracies)*100:.2f}%")
    scaler_finale = StandardScaler()
    X_scaled_finale = scaler_finale.fit_transform(X)
    grid_finale = GridSearchCV(SVC(probability=True), param_grid, cv=5, n_jobs=-1)
    grid_finale.fit(X_scaled_finale, y)
    joblib.dump({'model_type': 'svm', 'model': grid_finale.best_estimator_, 'scaler': scaler_finale}, SVM_MODEL_PATH)

def train_minirocket():
    print("="*60)
    print("[MINIROCKET] Caricamento e Preprocessing (Serie Temporali a 4 canali)...")
    X, y, subjects = load_dataset_minirocket()
    print(f"[MINIROCKET] Dataset pronto: {X.shape[0]} epoche valide da {len(np.unique(subjects))} soggetti.")
    
    logo = LeaveOneGroupOut()
    accuracies = []
    
    for train_index, test_index in logo.split(X, y, groups=subjects):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        minirocket = MiniRocketMultivariate()
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        X_train_transformed = minirocket.fit_transform(X_train)
        classifier.fit(X_train_transformed, y_train)
        preds = classifier.predict(minirocket.transform(X_test))
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        print(f"  -> Soggetto {subjects[test_index[0]]:2d} escluso | Accuratezza: {acc*100:5.2f}%")
    
    print(f"\n[MINIROCKET] Accuratezza Media Reale (Cross-Subject): {np.mean(accuracies)*100:.2f}%")
    minirocket_finale = MiniRocketMultivariate()
    X_transformed_finale = minirocket_finale.fit_transform(X)
    classifier_finale = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
    classifier_finale.fit(X_transformed_finale, y)
    joblib.dump({'model_type': 'minirocket', 'transformer': minirocket_finale, 'classifier': classifier_finale}, MINIROCKET_MODEL_PATH)

def train_eegnet():
    print("="*60)
    print("[EEGNET] Caricamento e Preprocessing (Tensori 4D per PyTorch)...")
    X, y, subjects = load_dataset_eegnet()
    print(f"[EEGNET] Dataset pronto: {X.shape[0]} epoche valide da {len(np.unique(subjects))} soggetti.")
    print(f"[EEGNET] Shape tensore di input: {X.shape}")
    
    logo = LeaveOneGroupOut()
    accuracies = []
    
    param_grid = {
        'module__dropoutRate': [0.25, 0.50]
    }
    
    print("[EEGNET] Avvio Validazione LOSO con Nested Grid Search (Attendere)...")
    
    for train_index, test_index in logo.split(X, y, groups=subjects):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Il wrapper di PyTorch
        net = NeuralNetBinaryClassifier(
            EEGNet,
            max_epochs=20,
            lr=0.005,
            optimizer=torch.optim.Adam,
            batch_size=32,
            train_split=None, 
            verbose=0
        )
        
        grid = GridSearchCV(net, param_grid, cv=2, n_jobs=1)
        grid.fit(X_train, y_train)
        
        preds = grid.predict(X_test)
        preds = (preds > 0.5).astype(int).flatten()
        
        acc = accuracy_score(y_test, preds)
        accuracies.append(acc)
        subject_id = subjects[test_index[0]]
        print(f"  -> Soggetto {subject_id:2d} escluso | Acc: {acc*100:5.2f}% | Best Param: {grid.best_params_}")
    
    print(f"\n[EEGNET] *** VALIDAZIONE LOSO COMPLETATA ***")
    print(f"[EEGNET] Accuratezza Media Reale (Cross-Subject): {np.mean(accuracies)*100:.2f}%")
    
    print("\n[EEGNET] Addestramento Finale su TUTTO il dataset...")
    net_finale = NeuralNetBinaryClassifier(
        EEGNet, max_epochs=20, lr=0.005, optimizer=torch.optim.Adam, batch_size=32, train_split=None, verbose=0
    )
    grid_finale = GridSearchCV(net_finale, param_grid, cv=3, n_jobs=1)
    grid_finale.fit(X, y)
    
    # In PyTorch salviamo i "pesi" della rete
    best_nn = grid_finale.best_estimator_.module_
    torch.save(best_nn.state_dict(), EEGNET_MODEL_PATH)
    
    print(f"[EEGNET] Modello salvato in: {EEGNET_MODEL_PATH}")
    print("="*60)

if __name__ == '__main__':
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', '..', 'models'), exist_ok=True)
    
    # Scegli cosa addestrare: 'svm', 'minirocket', o 'eegnet'
    MODEL_TYPE = 'eegnet' 
    
    if MODEL_TYPE == 'svm': train_svm()
    elif MODEL_TYPE == 'minirocket': train_minirocket()
    elif MODEL_TYPE == 'eegnet': train_eegnet()