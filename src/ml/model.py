import os
import numpy as np
import joblib

# La cartella base dove sono raccolti tutti i modelli
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

class AnxietyModel:
    def __init__(self):
        self.model_type = None
        self._pipeline = None
        self._loaded = False

    def load(self, model_type: str = 'svm'):
        # Costruiamo dinamicamente il nome del file da caricare
        file_name = f"{model_type}_model.pkl"
        model_path = os.path.join(MODELS_DIR, file_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f'File modello non trovato: {model_path}. '
                f'Assicurati di aver eseguito train.py impostando MODEL_TYPE = "{model_type}".'
            )
        
        self._pipeline = joblib.load(model_path)
        self.model_type = model_type
        self._loaded = True
        print(f"[ML] Caricato modello specifico: {file_name}")

    def predict_svm(self, tbr: float, faa: float) -> tuple[int, float]:
        if not self._loaded or self.model_type != 'svm':
            raise RuntimeError('Modello SVM non caricato correttamente.')
        
        scaler = self._pipeline['scaler']
        clf = self._pipeline['model']
        
        feat = scaler.transform([[tbr, faa]])
        label = int(clf.predict(feat)[0])
        prob = float(clf.predict_proba(feat)[0][1])
        return label, prob

    def predict_minirocket(self, raw_af7: list, raw_af8: list) -> tuple[int, float]:
        if not self._loaded or self.model_type != 'minirocket':
            raise RuntimeError('Modello MiniRocket non caricato correttamente.')
        
        transformer = self._pipeline['transformer']
        classifier = self._pipeline['classifier']
        
        X_realtime = np.array([[raw_af7, raw_af8]])
        X_transformed = transformer.transform(X_realtime)
        
        label = int(classifier.predict(X_transformed)[0])
        decision = classifier.decision_function(X_transformed)[0]
        prob = float(1 / (1 + np.exp(-decision))) 
        return label, prob