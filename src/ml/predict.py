import warnings
from src.ml.model import AnxietyModel

warnings.filterwarnings("ignore", category=UserWarning)

class AnxietyPredictor:
    def __init__(self, use_ml: bool = True, model_type: str = 'svm'):
        """
        model_type può essere: 'svm' oppure 'minirocket'
        """
        self.use_ml = use_ml
        self.model = AnxietyModel()
        self.label = None
        self.anxiety_prob = None

        if use_ml:
            try:
                # Passiamo il tipo di modello desiderato alla funzione load
                self.model.load(model_type)
            except Exception as e:
                print(f"[ML ERRORE] Impossibile caricare il modello richiesto ({model_type}): {e}")
                self.use_ml = False

    def predict(self, tbr: float = None, faa: float = None, raw_af7: list = None, raw_af8: list = None, anxiety_score: float = None):
        if self.use_ml:
            if self.model.model_type == 'svm' and tbr is not None and faa is not None:
                self.label, self.anxiety_prob = self.model.predict_svm(tbr, faa)
                
            elif self.model.model_type == 'minirocket' and raw_af7 is not None and raw_af8 is not None:
                self.label, self.anxiety_prob = self.model.predict_minirocket(raw_af7, raw_af8)
                
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