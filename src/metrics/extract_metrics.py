import numpy as np
from src.utils.utils import safe_log


class ExtractMetrics:
    def __init__(self):
        self.tbr = None    # Theta/Beta Ratio  — alto = calma
        self.faa = None    # Frontal Alpha Asymmetry — >0 calma, <0 ansia
        self.anxiety_score = None  # score normalizzato [0,1]

    def extract_metrics(self, band_buffer: list, band_buffer_size: int):
        if len(band_buffer) < band_buffer_size:
            return

        af7_theta = np.mean([e['AF7']['theta'] for e in band_buffer])
        af7_alpha = np.mean([e['AF7']['alpha'] for e in band_buffer])
        af7_beta  = np.mean([e['AF7']['beta']  for e in band_buffer])
        af8_alpha = np.mean([e['AF8']['alpha'] for e in band_buffer])

        # TBR: alto → calma/bassa ansia
        self.tbr = af7_theta / (af7_beta + 1e-10)

        # FAA: >0 → approccio/calma  |  <0 → evitamento/ansia
        self.faa = safe_log(af8_alpha) - safe_log(af7_alpha)

        # Anxiety score: combina TBR inverso e FAA negativo → [0=calma, 1=ansia]
        tbr_norm = np.clip(1.0 - (self.tbr / 3.0), 0.0, 1.0)
        faa_norm = np.clip((-self.faa + 1.0) / 2.0, 0.0, 1.0)
        self.anxiety_score = float(0.5 * tbr_norm + 0.5 * faa_norm)

        print(f'TBR: {self.tbr:.3f}  |  FAA: {self.faa:.3f}  |  Anxiety score: {self.anxiety_score:.3f}')
