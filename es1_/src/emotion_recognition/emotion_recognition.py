from src.emotion_recognition.load_models import LoadModels
from src.emotion_recognition.filter import Filter
from src.emotion_recognition.predict_emotion import PredictEmotion

class EmotionRecognition(LoadModels, Filter, PredictEmotion):
    def __init__(self):
        LoadModels.__init__(self)
        Filter.__init__(self)
        PredictEmotion.__init__(self)
