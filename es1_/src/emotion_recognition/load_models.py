import pickle as pkl
from glob import glob

class LoadModels:
    def __init__(self):
        self.models_excited = []
        self.models_relaxation = []
        self.models_sad = []
        self.models_angry = []

        # load models dict excited
        model_excited = pkl.load(open('models/emotion_model_label_10sec_1.pkl', 'rb'))
        for user in model_excited:
            self.models_excited.append(model_excited[user][4])

        # load models dict relaxation
        model_relaxation = pkl.load(open('models/emotion_model_label_10sec_2.pkl', 'rb'))
        for user in model_relaxation:
            self.models_relaxation.append(model_relaxation[user][4])

        # load model dict sad
        model_sad = pkl.load(open('models/emotion_model_label_10sec_3.pkl', 'rb'))
        for user in model_sad:
            self.models_sad.append(model_sad[user][4])

        # load model dict angry
        model_angry = pkl.load(open('models/emotion_model_label_10sec_4.pkl', 'rb'))
        for user in model_angry:
            self.models_angry.append(model_angry[user][4])
