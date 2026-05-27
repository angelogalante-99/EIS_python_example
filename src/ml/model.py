import os
import numpy as np
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'svm_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'scaler.pkl')


class AnxietyModel:
    def __init__(self):
        self._clf = None
        self._scaler = None
        self._loaded = False

    def load(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f'Modello non trovato: {MODEL_PATH}. '
                'Esegui prima src/ml/train.py con il dataset DREAMER.'
            )
        self._clf = joblib.load(MODEL_PATH)
        self._scaler = joblib.load(SCALER_PATH)
        self._loaded = True

    def predict(self, tbr: float, faa: float) -> tuple[int, float]:
        """
        Ritorna (label, probability):
          label 0 = calma, label 1 = ansia
          probability in [0, 1]
        """
        if not self._loaded:
            raise RuntimeError('Chiama load() prima di predict().')
        feat = self._scaler.transform([[tbr, faa]])
        label = int(self._clf.predict(feat)[0])
        prob = float(self._clf.predict_proba(feat)[0][1])
        return label, prob
