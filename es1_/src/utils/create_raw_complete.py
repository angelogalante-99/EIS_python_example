import mne

def create_raw_complete(EEG, sfreq):
    eeg = EEG * 1e-6
    ch_names = ['TP9', 'AF7', 'AF8', 'TP10']
    ch_types = ['eeg'] * len(ch_names)
    info = mne.create_info(ch_names, ch_types=ch_types, sfreq=sfreq)
    raw = mne.io.RawArray(eeg, info, verbose=0)
    raw.set_montage('standard_1020')

    return raw
