import pickle as pkl
from tqdm import tqdm
import numpy as np


class PredictEmotion:
    def __init__(self):
        self.anger_prediction = None
        self.sad_prediction = None
        self.relaxation_prediction = None
        self.excited_prediction = None

    def predict_emotion(self):
        prediction_excited = []
        prediction_relaxation = []
        prediction_sad = []
        prediction_anger = []
        # Previsione: Excited
        for model in tqdm(self.models_excited, desc="Predicting Excited"):
            prediction_excited.append(model.predict_proba(self.epoch_filtered_to_emotion))

        # Previsione: Relaxation
        for model in tqdm(self.models_relaxation, desc="Predicting Relaxation"):
            prediction_relaxation.append(model.predict_proba(self.epoch_filtered_to_emotion))

        # Previsione: Sad
        for model_sad in tqdm(self.models_sad, desc="Predicting Sad"):
            prediction_sad.append(model_sad.predict_proba(self.epoch_filtered_to_emotion))

        # Previsione: Angry
        for model_angry in tqdm(self.models_angry, desc="Predicting Angry"):
            prediction_anger.append(model_angry.predict_proba(self.epoch_filtered_to_emotion))

        self.excited_prediction = np.array(prediction_excited).mean(axis=0)
        print(f'excited prediction: {self.excited_prediction}')
        self.relaxation_prediction = np.array(prediction_relaxation).mean(axis=0)
        print(f'relaxation prediction: {self.relaxation_prediction}')
        self.sad_prediction = np.array(prediction_sad).mean(axis=0)
        print(f'sad prediction: {self.sad_prediction}')
        self.anger_prediction = np.array(prediction_anger).mean(axis=0)
        print(f'anger prediction: {self.anger_prediction}')

