"""
Player per suoni IADS (International Affective Digitized Sounds).

Posiziona i file audio (WAV) in data/iads/ prima di eseguire.
Selezione: basso Arousal + alta Valence (pioggia, acqua, natura).
"""
import os
import random
import pygame

IADS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'iads')
CALM_VOLUME = 0.6
ANXIETY_VOLUME_FACTOR = 0.8  # riduci volume quando ansia alta


class IADSPlayer:
    def __init__(self):
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self._sounds: list[str] = []
        self._current: pygame.mixer.Sound | None = None
        self._channel: pygame.mixer.Channel | None = None
        self._load_sounds()

    def _load_sounds(self):
        if not os.path.isdir(IADS_DIR):
            print(f'[IADSPlayer] Cartella IADS non trovata: {IADS_DIR}')
            return
        self._sounds = [
            os.path.join(IADS_DIR, f)
            for f in os.listdir(IADS_DIR)
            if f.lower().endswith('.wav')
        ]
        if not self._sounds:
            print(f'[IADSPlayer] Nessun file WAV trovato in {IADS_DIR}')

    def start(self):
        if not self._sounds:
            return
        path = random.choice(self._sounds)
        self._current = pygame.mixer.Sound(path)
        self._channel = self._current.play(loops=-1)
        self._channel.set_volume(CALM_VOLUME)

    def update_volume(self, anxiety_prob: float):
        if self._channel:
            vol = CALM_VOLUME * (1.0 - anxiety_prob * ANXIETY_VOLUME_FACTOR)
            self._channel.set_volume(max(0.1, vol))

    def stop(self):
        if self._channel:
            self._channel.stop()
        pygame.mixer.quit()
