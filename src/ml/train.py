"""
Addestramento SVM sul dataset Muse (BIDS) con LOSO cross-validation.
VERSIONE IBRIDA: Logica Baseline (2 Feature) + Epoching a 4 Secondi.

Struttura attesa:
  data/dataframe.pkl  — generato da data_extraction.py
  data/Video_selezionati.xlsx — mappatura Experiment_id → VAQ label
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib

DATA_PKL   = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dataframe.pkl')
LABELS_XLS = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Video_selezionati.xlsx')
MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'svm_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'scaler.pkl')

# Canali nel segnale Muse 2
AF7_IDX = 0
AF8_IDX = 3
SFREQ = 256

# Bande EEG in Hz
THETA = (4, 8)
ALPHA = (8, 13)
BETA  = (13, 30)

# Sessioni con dati corrotti da ignorare
CORRUPTED = {('sub-ID021', 'ses-S023')}
# Sessioni con sampling rate anomalo (ID015 S033, S038) — incluse ma gestite
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
    return int(session_id.replace('ses-S', '').lstrip('0') or '0')


def _band_power(signal: np.ndarray, low: float, high: float, fs: int) -> float:
    freqs, psd = welch(signal, fs=fs, nperseg=min(fs * 2, len(signal)))
    idx = np.logical_and(freqs >= low, freqs <= high)
    return float(np.trapz(psd[idx], freqs[idx]))


def _extract_features(time_series: np.ndarray, fs: int = SFREQ) -> np.ndarray:
    """Estrae TBR e FAA dall'epoca di 4 secondi passata."""
    af7 = time_series[AF7_IDX]
    af8 = time_series[AF8_IDX]

    af7_theta = _band_power(af7, *THETA, fs)
    af7_alpha = _band_power(af7, *ALPHA, fs)
    af7_beta  = _band_power(af7, *BETA,  fs)
    af8_alpha = _band_power(af8, *ALPHA, fs)

    tbr = af7_theta / (af7_beta + 1e-10)
    faa = np.log(af8_alpha + 1e-10) - np.log(af7_alpha + 1e-10)
    return np.array([tbr, faa])


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(DATA_PKL, 'rb') as f:
        data = pickle.load(f)

    label_map = _build_label_map(LABELS_XLS)

    X, y, subjects = [], [], []
    subject_list = sorted(data.keys())

    for subj_idx, subj_name in enumerate(subject_list):
        for session in data[subj_name]:
            sid = session['session_id']

            if (subj_name, sid) in CORRUPTED:
                continue

            exp_id = _session_to_exp_id(sid)
            if exp_id not in label_map:
                continue

            ts = session['time_series']  # shape (4, N)
            fs = LOW_SFREQ.get((subj_name, sid), SFREQ)
            
            # --- MODIFICA CHIAVE: EPOCHING A 4 SECONDI ---
            step = 4 * fs  # 4 secondi (es. 1024 campioni a 256Hz)
            n_samples = ts.shape[1]
            
            # Scorre l'intera registrazione a blocchi di 4 secondi
            for i in range(0, n_samples - step, step):
                epoch_data = ts[:, i:i+step]
                
                feat = _extract_features(epoch_data, fs=fs)
                
                if not np.isnan(feat).any():
                    X.append(feat)
                    y.append(label_map[exp_id])
                    subjects.append(subj_idx)
            # ---------------------------------------------

    return np.array(X), np.array(y), np.array(subjects)


def loso_cross_validation() -> float:
    X, y, subjects = load_dataset()
    print(f'Dataset: {len(X)} epoche da 4s, {len(np.unique(subjects))} soggetti')
    print(f'Distribuzione label: calma={np.sum(y==0)}, ansia={np.sum(y==1)}\n')

    unique_subjects = np.unique(subjects)
    accuracies = []

    for test_subj in unique_subjects:
        train_mask = subjects != test_subj
        test_mask  = subjects == test_subj

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_mask])
        X_test  = scaler.transform(X[test_mask])

        clf = SVC(kernel='rbf', C=1.0, gamma='scale')
        clf.fit(X_train, y[train_mask])
        preds = clf.predict(X_test)
        acc = accuracy_score(y[test_mask], preds)
        accuracies.append(acc)

    mean_acc = float(np.mean(accuracies))
    print(f'\nLOSO mean accuracy: {mean_acc:.3f}')
    return mean_acc


def train_final_model():
    X, y, _ = load_dataset()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf.fit(X_scaled, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f'\nModello definitivo e Scaler salvati in {os.path.dirname(MODEL_PATH)}')


if __name__ == '__main__':
    print("Avvio Cross-Validazione LOSO sulle epoche...")
    loso_cross_validation()
    print("\nAvvio Addestramento Finale...")
    train_final_model()