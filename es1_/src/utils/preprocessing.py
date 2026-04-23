
import numpy
from scipy import signal
from scipy.signal import iirnotch, medfilt


def butter_bandpass(lowcut, highcut, fs, filter_order=4):
    """
    Design Butterworth bandpass filter coefficients.

    :param lowcut: Low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz)
    :type highcut: float
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param filter_order: Filter order (default: 4)
    :type filter_order: int
    :return: Numerator (b) and denominator (a) polynomials
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(filter_order, [low, high], btype='band')
    return b, a


def chebyshev_bandpass(lowcut, highcut, fs, filter_order=4, rp=1):
    """
    Design Chebyshev Type I bandpass filter coefficients.

    :param lowcut: Low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz)
    :type highcut: float
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param filter_order: Filter order (default: 4)
    :type filter_order: int
    :param rp: Peak-to-peak ripple (dB)
    :type rp: float
    :return: Numerator (b) and denominator (a) polynomials
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.cheby1(filter_order, rp, [low, high], btype='band')
    return b, a


def bessel_bandpass(lowcut, highcut, fs, filter_order=4):
    """
    Design Bessel bandpass filter coefficients.

    :param lowcut: Low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz)
    :type highcut: float
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param filter_order: Filter order (default: 4)
    :type filter_order: int
    :return: Numerator (b) and denominator (a) polynomials
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.bessel(filter_order, [low, high], btype='band')
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply Butterworth lowpass filter to signal data.

    :param data: Input signal data
    :type data: numpy.ndarray
    :param cutoff: Cutoff frequency (Hz)
    :type cutoff: float
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param order: Filter order (default: 4)
    :type order: int
    :return: Filtered signal data
    :rtype: numpy.ndarray
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    y = signal.filtfilt(b, a, data)
    return y


def signal_filter(data, fs, lowcut, highcut, filter_order=4, method='Butterworth'):
    """
    Apply bandpass filter to signal data.

    :param data: Input signal data
    :type data: numpy.ndarray
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param lowcut: Low cutoff frequency (Hz)
    :type lowcut: float
    :param highcut: High cutoff frequency (Hz)
    :type highcut: float
    :param filter_order: Filter order (default: 4)
    :type filter_order: int
    :param method: Filter design method (default: 'Butterworth')
    :type method: str
    :return: Filtered signal data
    :rtype: numpy.ndarray
    """
    if method == 'Butterworth':
        b, a = butter_bandpass(lowcut, highcut, fs, filter_order)
    elif method == 'Bessel':
        b, a = bessel_bandpass(lowcut, highcut, fs, filter_order)
    elif method == 'chebyshev':
        b, a = chebyshev_bandpass(lowcut, highcut, fs, filter_order, rp=1)
    filter_data = signal.filtfilt(b, a, data, padlen=200)
    return filter_data


def notch_filter(data, fs, f0=50, Q=30):
    """
    Apply notch filter to remove powerline interference.

    :param data: Input signal data
    :type data: numpy.ndarray
    :param fs: Sampling frequency (Hz)
    :type fs: float
    :param f0: Notch frequency (default: 50 Hz)
    :type f0: float
    :param Q: Quality factor (default: 30)
    :type Q: float
    :return: Filtered signal data
    :rtype: numpy.ndarray
    """
    b, a = iirnotch(f0, Q, fs)
    filter_data = signal.filtfilt(b, a, data, padlen=200)
    return filter_data







