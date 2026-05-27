from src.audio.binaural import BinauralBeats
from src.audio.iads_player import IADSPlayer


class AudioEngine:
    def __init__(self, use_binaural: bool = False, use_iads: bool = True):
        self._use_binaural = use_binaural
        self._use_iads = use_iads
        self._binaural = BinauralBeats() if use_binaural else None
        self._iads = IADSPlayer("data/iads") if use_iads else None

    def start(self):
        if self._binaural:
            self._binaural.start()
        if self._iads:
            self._iads.start()

    def update(self, anxiety_prob: float):
        if self._binaural:
            self._binaural.set_anxiety_level(anxiety_prob)
            self._binaural.play_chunk()
        if self._iads:
            self._iads.set_anxiety_level(anxiety_prob)

    def stop(self):
        if self._binaural:
            self._binaural.stop()
        if self._iads:
            self._iads.stop()
