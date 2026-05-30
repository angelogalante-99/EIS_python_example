import warnings
from src.ml.model import AnxietyModel

warnings.filterwarnings("ignore", category=UserWarning)

class AnxietyPredictor:
    def __init__(self, use_ml: bool = True, model_type: str = 'svm'):
        self.use_ml = use_ml
        self.model = AnxietyModel()
        self.label = None
        self.anxiety_prob = None

        if use_ml:
            try:
                self.model.load(model_type)
            except Exception as e:
                print(f"[ML ERRORE] Impossibile caricare il modello ({model_type}): {e}")
                self.use_ml = False

    def predict(self, 
                tbr_af7: float = None, tbr_tp9: float = None, tbr_tp10: float = None, tbr_af8: float = None, 
                faa: float = None, taa: float = None, 
                raw_af7: list = None, raw_tp9: list = None, raw_tp10: list = None, raw_af8: list = None, 
                anxiety_score: float = None):
        if self.use_ml:
            # Controllo 6 feature per SVM
            if self.model.model_type == 'svm' and all(v is not None for v in [tbr_af7, tbr_tp9, tbr_tp10, tbr_af8, faa, taa]):
                self.label, self.anxiety_prob = self.model.predict_svm(tbr_af7, tbr_tp9, tbr_tp10, tbr_af8, faa, taa)
                
            # Controllo 4 canali RAW per MiniRocket
            elif self.model.model_type == 'minirocket' and all(v is not None for v in [raw_af7, raw_tp9, raw_tp10, raw_af8]):
                self.label, self.anxiety_prob = self.model.predict_minirocket(raw_af7, raw_tp9, raw_tp10, raw_af8)
                
            # Controllo 4 canali RAW per EEGNet
            elif self.model.model_type == 'eegnet' and all(v is not None for v in [raw_af7, raw_tp9, raw_tp10, raw_af8]):
                self.label, self.anxiety_prob = self.model.predict_eegnet(raw_af7, raw_tp9, raw_tp10, raw_af8)
                
            else:
                self.anxiety_prob = anxiety_score or 0.5
                self.label = 1 if self.anxiety_prob > 0.5 else 0
        else:
            self.anxiety_prob = anxiety_score or 0.0
            self.label = 1 if self.anxiety_prob > 0.5 else 0

        state = 'ANSIA' if self.label == 1 else 'CALMA'
        tipo_modello = self.model.model_type.upper() if self.use_ml else "MANUALE"
        print(f'Stato [{tipo_modello}]: {state} (prob: {self.anxiety_prob:.3f})')
        return self.label, self.anxiety_prob