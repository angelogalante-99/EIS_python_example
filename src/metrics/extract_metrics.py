import numpy as np
from src.utils.utils import safe_log

class ExtractMetrics:
    def __init__(self):
        self.tbr_af7 = None
        self.tbr_tp9 = None
        self.tbr_tp10 = None
        self.tbr_af8 = None
        self.faa = None
        self.taa = None
        self.anxiety_score = None 

    def extract_metrics(self, band_buffer: list, band_buffer_size: int):
        if len(band_buffer) < band_buffer_size:
            return

        af7_theta = np.mean([e['AF7']['theta'] for e in band_buffer])
        af7_alpha = np.mean([e['AF7']['alpha'] for e in band_buffer])
        af7_beta  = np.mean([e['AF7']['beta']  for e in band_buffer])
        
        tp9_theta = np.mean([e['TP9']['theta'] for e in band_buffer])
        tp9_alpha = np.mean([e['TP9']['alpha'] for e in band_buffer])
        tp9_beta  = np.mean([e['TP9']['beta']  for e in band_buffer])
        
        tp10_theta = np.mean([e['TP10']['theta'] for e in band_buffer])
        tp10_alpha = np.mean([e['TP10']['alpha'] for e in band_buffer])
        tp10_beta  = np.mean([e['TP10']['beta']  for e in band_buffer])
        
        af8_theta = np.mean([e['AF8']['theta'] for e in band_buffer])
        af8_alpha = np.mean([e['AF8']['alpha'] for e in band_buffer])
        af8_beta  = np.mean([e['AF8']['beta']  for e in band_buffer])

        # Calcolo TBR su tutti e 4 i canali
        self.tbr_af7 = af7_theta / (af7_beta + 1e-10)
        self.tbr_tp9 = tp9_theta / (tp9_beta + 1e-10)
        self.tbr_tp10 = tp10_theta / (tp10_beta + 1e-10)
        self.tbr_af8 = af8_theta / (af8_beta + 1e-10)

        # Asimmetrie (Frontale e Temporale)
        self.faa = safe_log(af8_alpha) - safe_log(af7_alpha)
        self.taa = safe_log(tp10_alpha) - safe_log(tp9_alpha)

        # Anxiety score manuale (basato sui frontali per coerenza col passato)
        tbr_norm = np.clip(1.0 - (self.tbr_af7 / 3.0), 0.0, 1.0)
        faa_norm = np.clip((-self.faa + 1.0) / 2.0, 0.0, 1.0)
        self.anxiety_score = float(0.5 * tbr_norm + 0.5 * faa_norm)