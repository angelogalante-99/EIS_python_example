class Filter:
    def __init__(self):
        self.raw = None

    def notch_filter(self, raw, notch_frequency: float):
        self.raw = raw.notch_filter(notch_frequency, verbose=False)

    def band_pass_filter(self, raw, low: float, high: float):
        self.raw = raw.filter(low, high, verbose=False)
