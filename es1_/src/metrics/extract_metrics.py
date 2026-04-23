import numpy as np
import pandas as pd


class ExtractMetrics:
    def __init__(self):
        self.valence = None
        self.FAA_2 = None
        self.FAA_1 = None
        self.FAA = None
        self.TP10_dominance = None
        self.TP9_dominance = None
        self.AF8_dominance = None
        self.AF7_dominance = None
        self.stress = None
        self.dominance = None
        self.engagement = None
        self.arousal = None
        self.TP10_relaxation = None
        self.TP10_focus = None
        self.TP9_relaxation = None
        self.TP9_focus = None
        self.AF8_relaxation = None
        self.AF8_focus = None
        self.AF7_relaxation = None
        self.AF7_focus = None
        self.focus = None
        self.relaxation = None

    def extract_metrics(self, band_buffer, band_buffer_size):
        if len(band_buffer) == band_buffer_size:
            band_mean_ = pd.concat(band_buffer).reset_index(drop=True)
            band_mean_.set_index('Channel', inplace=True)

            channels = band_mean_.index.unique()
            channel_band_values = {ch: band_mean_.loc[ch] for ch in channels}

            TP9_delta = channel_band_values['TP9']['Delta'].mean()
            TP9_theta = channel_band_values['TP9']['Theta'].mean()
            TP9_alpha = channel_band_values['TP9']['Alpha'].mean()
            TP9_beta = channel_band_values['TP9']['Beta'].mean()

            AF7_delta = channel_band_values['AF7']['Delta'].mean()
            AF7_theta = channel_band_values['AF7']['Theta'].mean()
            AF7_alpha = channel_band_values['AF7']['Alpha'].mean()
            AF7_beta = channel_band_values['AF7']['Beta'].mean()

            AF8_delta = channel_band_values['AF8']['Delta'].mean()
            AF8_theta = channel_band_values['AF8']['Theta'].mean()
            AF8_alpha = channel_band_values['AF8']['Alpha'].mean()
            AF8_beta = channel_band_values['AF8']['Beta'].mean()

            TP10_delta = channel_band_values['TP10']['Delta'].mean()
            TP10_theta = channel_band_values['TP10']['Theta'].mean()
            TP10_alpha = channel_band_values['TP10']['Alpha'].mean()
            TP10_beta = channel_band_values['TP10']['Beta'].mean()

            self.relaxation = AF7_alpha / AF7_delta # valori altri = buono stato di rilassamento cosciente (Alpha protocol)
            self.focus = AF7_beta / AF7_theta # valori alti = concentrazione alta
            self.stress = AF7_theta / AF7_alpha # valori alti = indicano meno stress
            self.engagement = AF7_beta / (AF7_alpha + AF7_theta) # valori alti indicano maggiore engagement
            self.arousal = (AF7_beta + AF8_beta) / (AF7_alpha + AF7_alpha) # valori alti = maggior arousal
            self.valence = (AF8_alpha/AF8_beta) - (AF7_alpha / AF7_beta) # polarità emotiva = 2 quadranti
            self.FAA = np.log(AF8_alpha / AF7_alpha) # attivazione cerebrale emisfero destro o sinistro

            print(f'relaxation: {self.relaxation:.2f}')
            print(f'focus level: {self.focus:.2f}')
            print(f'stress level: {self.stress:.2f}')
            print(f'arousal index: {self.arousal:.2f}')
            print(f'valence index: {self.valence:.2f}')
            print(f'FAA index: {self.FAA:.2f}')


