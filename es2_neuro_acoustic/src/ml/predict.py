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
        if self.use_ml and tbr is not None and faa is not None:
            self.label, self.anxiety_prob = self.model.predict(tbr, faa)
        else:
            # Fallback a soglie deterministiche
            self.anxiety_prob = anxiety_score or 0.0
            self.label = 1 if self.anxiety_prob > 0.5 else 0

        state = 'ANSIA' if self.label == 1 else 'CALMA'
        print(f'Stato: {state}  (prob ansia: {self.anxiety_prob:.3f})')
        return self.label, self.anxiety_prob
