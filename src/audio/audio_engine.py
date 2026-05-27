from src.audio.soundscapes_player import soundscapesplayer

class AudioEngine:
    def __init__(self, use_soundscapes: bool = True):
        self._use_iads = use_soundscapes
        self._iads = soundscapesplayer() if use_soundscapes else None

    def start(self):
        if self._iads:
            self._iads.start()

    def update(self, anxiety_prob: float):
        if self._iads:
            # Passa la palla a soundscapesplayer
            self._iads.update_volume(anxiety_prob)

    def stop(self):
        if self._iads:
            self._iads.stop()