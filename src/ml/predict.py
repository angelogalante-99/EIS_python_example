from src.ml.model import AnxietyModel


class AnxietyPredictor:
    def __init__(self, use_ml: bool = True):
        self.use_ml = use_ml
        self.model = AnxietyModel()
        self.label = None
        self.anxiety_prob = None

        if use_ml:
            self.model.load()

    def predict(self, tbr: float, faa: float, anxiety_score: float = None):
        # 1. Calcolo della probabilità (tramite Machine Learning o Fallback)
        if self.use_ml and tbr is not None and faa is not None:
            # Estraiamo solo la probabilità, ignorando l'etichetta "standard" del modello
            _, self.anxiety_prob = self.model.predict(tbr, faa)
        else:
            # Fallback a valori deterministici
            self.anxiety_prob = anxiety_score or 0.0

        # 2. Applichiamo la nostra nuova SOGLIA MASTER (0.75) a tutto il sistema
        self.label = 1 if self.anxiety_prob >= 0.75 else 0

        # 3. Testo a schermo sincronizzato al 100% con la musica
        state = 'ANSIA' if self.label == 1 else 'CALMA'
        print(f'Stato: {state}  (prob ansia: {self.anxiety_prob:.3f})')

        return self.label, self.anxiety_prob