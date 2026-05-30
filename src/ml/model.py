import os
import numpy as np
import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'models')

# ==========================================
# LA RETE NEURALE PYTORCH (EEGNET)
# ==========================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, chans=4, samples=1024, dropoutRate=0.5, kernLength=128, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        # 1. Temporal Convolution (Filtro frequenze)
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # 2. Spatial Convolution (Asimmetrie spaziali)
        self.depthwise1 = nn.Conv2d(F1, F1 * D, (chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # 3. Separable Convolution (Riassunto features)
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding='same', groups=F1 * D, bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # Classificatore Lineare finale
        # 1024 / 4 / 8 = 32. Moltiplicato per F2 (16) = 512
        self.fc1 = nn.Linear(F2 * 32, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Appiattisce il tensore per passarlo all'ultimo strato
        x = x.view(x.size(0), -1) 
        x = self.fc1(x)
        return x # Il Sigmoid viene applicato dopo

# ==========================================
# IL MANAGER DEI MODELLI
# ==========================================
class AnxietyModel:
    def __init__(self):
        self.model_type = None
        self._pipeline = None
        self._pytorch_model = None
        self._loaded = False

    def load(self, model_type: str = 'svm'):
        self.model_type = model_type
        
        if model_type == 'eegnet':
            file_name = f"{model_type}_model.pt"
            model_path = os.path.join(MODELS_DIR, file_name)
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f'Modello non trovato: {model_path}')
            
            # Istanzia la rete e carica i "pesi" salvati
            self._pytorch_model = EEGNet()
            self._pytorch_model.load_state_dict(torch.load(model_path))
            self._pytorch_model.eval() # Imposta la rete in modalità "Inferenza" (spegne il dropout)
            self._loaded = True
            print(f"[ML] Caricato modello PyTorch Deep Learning: {file_name}")
            return

        file_name = f"{model_type}_model.pkl"
        model_path = os.path.join(MODELS_DIR, file_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f'Modello non trovato: {model_path}')
        
        self._pipeline = joblib.load(model_path)
        self._loaded = True
        print(f"[ML] Caricato modello specifico: {file_name}")

    def predict_svm(self, tbr_af7: float, tbr_tp9: float, tbr_tp10: float, tbr_af8: float, faa: float, taa: float) -> tuple[int, float]:
        if not self._loaded or self.model_type != 'svm':
            raise RuntimeError('Modello SVM non caricato.')
        scaler = self._pipeline['scaler']
        clf = self._pipeline['model']
        feat = scaler.transform([[tbr_af7, tbr_tp9, tbr_tp10, tbr_af8, faa, taa]])
        label = int(clf.predict(feat)[0])
        prob = float(clf.predict_proba(feat)[0][1])
        return label, prob

    def predict_minirocket(self, raw_af7: list, raw_tp9: list, raw_tp10: list, raw_af8: list) -> tuple[int, float]:
        if not self._loaded or self.model_type != 'minirocket':
            raise RuntimeError('Modello MiniRocket non caricato.')
        transformer = self._pipeline['transformer']
        classifier = self._pipeline['classifier']
        X_realtime = np.array([[raw_af7, raw_tp9, raw_tp10, raw_af8]])
        X_transformed = transformer.transform(X_realtime)
        label = int(classifier.predict(X_transformed)[0])
        decision = classifier.decision_function(X_transformed)[0]
        prob = float(1 / (1 + np.exp(-decision))) 
        return label, prob

    def predict_eegnet(self, raw_af7: list, raw_tp9: list, raw_tp10: list, raw_af8: list) -> tuple[int, float]:
        if not self._loaded or self.model_type != 'eegnet':
            raise RuntimeError('Modello EEGNet non caricato.')
        
        # PyTorch si aspetta [Batch, CanaleImg, Canali, Samples]
        X_realtime = np.array([[raw_af7, raw_tp9, raw_tp10, raw_af8]], dtype=np.float32)
        X_realtime = X_realtime[:, np.newaxis, :, :] # Diventa (1, 1, 4, 1024)
        
        with torch.no_grad(): # Disattiva il calcolo dei gradienti per massima velocità
            tensor_x = torch.from_numpy(X_realtime)
            raw_output = self._pytorch_model(tensor_x)
            prob = float(torch.sigmoid(raw_output).item())
            
        label = 1 if prob > 0.5 else 0
        return label, prob