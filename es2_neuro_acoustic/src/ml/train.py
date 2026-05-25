"""
Addestramento SVM su DREAMER con LOSO cross-validation.

Dataset DREAMER: https://zenodo.org/record/546113
Scarica il file DREAMER.mat e posizionalo in data/DREAMER.mat prima di eseguire.
"""
import numpy as np
import scipy.io
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'DREAMER.mat')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'svm_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'scaler.pkl')

# Canali AF3/AF4 nel dataset DREAMER (equivalenti a AF7/AF8 del Muse 2)
AF3_IDX = 0
AF4_IDX = 1

THETA = (4, 8)
ALPHA = (8, 13)
BETA = (13, 30)
SFREQ = 128  # DREAMER usa 128 Hz


def _band_power(signal: np.ndarray, low: float, high: float, fs: int) -> float:
    from scipy.signal import welch
    freqs, psd = welch(signal, fs=fs, nperseg=fs * 2)
    idx = np.logical_and(freqs >= low, freqs <= high)
    return float(np.trapz(psd[idx], freqs[idx]))


def _extract_features(eeg_epoch: np.ndarray, fs: int = SFREQ) -> np.ndarray:
    """Estrae TBR e FAA da un'epoca EEG (shape: samples x channels)."""
    af3 = eeg_epoch[:, AF3_IDX]
    af4 = eeg_epoch[:, AF4_IDX]

    af3_theta = _band_power(af3, *THETA, fs)
    af3_alpha = _band_power(af3, *ALPHA, fs)
    af3_beta  = _band_power(af3, *BETA,  fs)
    af4_alpha = _band_power(af4, *ALPHA, fs)

    tbr = af3_theta / (af3_beta + 1e-10)
    faa = np.log(af4_alpha + 1e-10) - np.log(af3_alpha + 1e-10)
    return np.array([tbr, faa])


def load_dreamer_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carica DREAMER.mat e restituisce (features, labels_arousal, subject_ids).
    Label binarizzata: arousal >= 3 → 1 (ansia/alta attivazione), < 3 → 0 (calma).
    """
    mat = scipy.io.loadmat(DATA_PATH, simplify_cells=True)
    dreamer = mat['DREAMER']
    data_list = dreamer['Data']

    X, y, subjects = [], [], []
    for subj_idx, subj in enumerate(data_list):
        for trial_idx in range(18):
            eeg = subj['EEG']['stimuli'][trial_idx]  # shape: (samples, 14)
            arousal = int(subj['ScoreArousal'][trial_idx])
            label = 1 if arousal >= 3 else 0
            feat = _extract_features(eeg)
            X.append(feat)
            y.append(label)
            subjects.append(subj_idx)

    return np.array(X), np.array(y), np.array(subjects)


def loso_cross_validation() -> float:
    """Leave-One-Subject-Out CV. Ritorna l'accuracy media."""
    X, y, subjects = load_dreamer_features()
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
        print(f'Subject {test_subj:02d} → accuracy: {acc:.3f}')

    mean_acc = float(np.mean(accuracies))
    print(f'\nLOSO mean accuracy: {mean_acc:.3f}')
    return mean_acc


def train_final_model():
    """Addestra il modello finale su tutti i soggetti e lo salva."""
    X, y, _ = load_dreamer_features()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
    clf.fit(X_scaled, y)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(clf, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f'Modello salvato in {MODEL_PATH}')


if __name__ == '__main__':
    loso_cross_validation()
    train_final_model()
