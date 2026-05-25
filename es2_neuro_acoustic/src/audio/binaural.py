"""
Generatore di binaural beats.

Come funziona: orecchio sinistro riceve frequenza base F,
orecchio destro riceve F + delta. Il cervello percepisce una
pulsazione alla frequenza delta (onda theta/alpha per calma).
"""
import numpy as np
import pyaudio

SAMPLE_RATE = 44100
CHUNK = 1024
BASE_FREQ = 200.0  # Hz — portante
CALM_BEAT = 6.0    # Hz — delta theta per rilassamento
ANXIETY_BEAT = 10.0  # Hz — delta alpha, leggermente più stimolante


class BinauralBeats:
    def __init__(self):
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._phase_left = 0.0
        self._phase_right = 0.0
        self._beat_freq = CALM_BEAT
        self._volume = 0.3

    def set_anxiety_level(self, anxiety_prob: float):
        # Interpola la frequenza di battito in base al livello di ansia
        self._beat_freq = CALM_BEAT + (ANXIETY_BEAT - CALM_BEAT) * anxiety_prob
        # Abbassa il volume proporzionalmente all'ansia per non sovraccaricare
        self._volume = 0.4 - 0.15 * anxiety_prob

    def _generate_chunk(self) -> bytes:
        t = np.arange(CHUNK) / SAMPLE_RATE
        left  = np.sin(2 * np.pi * BASE_FREQ * t + self._phase_left)
        right = np.sin(2 * np.pi * (BASE_FREQ + self._beat_freq) * t + self._phase_right)

        self._phase_left  = (self._phase_left  + 2 * np.pi * BASE_FREQ * CHUNK / SAMPLE_RATE) % (2 * np.pi)
        self._phase_right = (self._phase_right + 2 * np.pi * (BASE_FREQ + self._beat_freq) * CHUNK / SAMPLE_RATE) % (2 * np.pi)

        stereo = np.empty(CHUNK * 2, dtype=np.float32)
        stereo[0::2] = (left  * self._volume).astype(np.float32)
        stereo[1::2] = (right * self._volume).astype(np.float32)
        return stereo.tobytes()

    def start(self):
        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=2,
            rate=SAMPLE_RATE,
            output=True,
            frames_per_buffer=CHUNK,
        )

    def play_chunk(self):
        if self._stream and self._stream.is_active():
            self._stream.write(self._generate_chunk())

    def stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        self._pa.terminate()
