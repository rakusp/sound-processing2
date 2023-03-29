import numpy as np


def stride_trick(a, stride_length, stride_step):
    """
    apply framing using the stride trick from numpy.

    Args:
        a (array) : signal array.
        stride_length (int) : length of the stride.
        stride_step (int) : stride step.

    Returns:
        blocked/framed array.
    """
    nrows = ((a.size - stride_length) // stride_step) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a,
                                           shape=(nrows, stride_length),
                                           strides=(stride_step * n, n))


def framing(sig, fs, win_len=0.025, win_hop=0.01):
    """
    transform a signal into a series of overlapping frames (=Frame blocking).

    Args:
        sig     (array) : a mono audio signal (Nx1) from which to compute features.
        fs        (int) : the sampling frequency of the signal we are working with.
                          Default is 16000.
        win_len (float) : window length in sec.
                          Default is 0.025.
        win_hop (float) : step between successive windows in sec.
                          Default is 0.01.

    Returns:
        array of frames.
        frame length.

    Notes:
    ------
        Uses the stride trick to accelerate the processing.
    """
    # run checks and assertions
    if win_len < win_hop:
        print("ParameterError: win_len must be larger than win_hop.")

    # compute frame length and frame step (convert from seconds to samples)
    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step

    # compute number of frames and left sample in order to pad if needed to make
    # sure all frames have equal number of samples  without truncating any samples
    # from the original signal
    rest_samples = np.abs(signal_length - frames_overlap) % np.abs(frame_length - frames_overlap)
    pad_signal = np.append(sig, np.array([0] * int(frame_step - rest_samples) * int(rest_samples != 0.)))

    # apply stride trick
    frames = stride_trick(pad_signal, int(frame_length), int(frame_step))
    return frames, frame_length


def short_time_energy(data):
    """
    Compute short time energy parameter for single signal frame.

    Args:
        data (array) : single signal frame

    Returns:
        Short time energy
    """
    return (1 / len(data)) * (np.sum(np.power(data, 2)))


def volume(data):
    """
    Compute volume parameter for single signal frame.

    Args:
        data (array) : single signal frame

    Returns:
        Volume
    """
    return np.sqrt(short_time_energy(data))


def zero_crossing_rate(data):
    """
    Compute zero crossing rate (ZCR) parameter for single signal frame.

    Args:
        data (array) : single signal frame

    Returns:
        Zero crossing rate
    """
    n = len(data)
    return (1 / (2 * n)) * (np.sum(np.abs(np.sign(data[1:n]) - np.sign(data[0:n - 1]))))


def autocorrelation_function(data, lag):
    """
    Compute autocorrelation function for single signal frame.

    Args:
        data (array) : single signal frame
        lag (int) : lag number

    Returns:
        Autocorrelation function
    """
    n = len(data)
    return np.sum(np.multiply((data[lag:n]), (data[0:n - lag])))


def average_magnitude_difference(data, lag):
    """
    Compute average magnitude difference parameter for single signal frame.

    Args:
        data (array) : single signal frame
        lag (int) : lag number

    Returns:
        Average magnitude difference
    """
    n = len(data)
    return np.sum(np.abs(np.sign(data[lag:n]) - np.sign(data[0:n - lag])))


def scale_data(data):
    """
    Scale given signal data to range [-1,1] based on the biggest absolute value of signal.

    Args:
        data (array) : one dimensional signal

    Returns:
        Scaled siganl data
    """
    min_val = np.min(data)
    max_val = np.max(data)
    scale = max(abs(np.min(data)), abs(np.max(data)))
    return np.divide(data, scale)

def detect_silence(data, vol_max):
    """
    Detects silent for given signal data based on average abs value.

    Args:
        data (array) : one dimensional signal

    Returns:
        Boolean which indicate whether silence was detected 
    """
    mean_vol = np.sum(np.abs(data))/len(data)
    if mean_vol > vol_max:
        return False
    else:
        return True
    
def low_short_time_energy_ratio(frames):
    ste = np.apply_along_axis(short_time_energy, 1, frames)
    return 1/(2*frames.shape[0]) * np.sum(np.sign(0.5*np.mean(ste) - ste)+1 )

def high_zero_crossing_rate_ratio(frames):
    zcr = np.apply_along_axis(zero_crossing_rate, 1, frames)
    return 1/(2*frames.shape[0]) * np.sum(np.sign(zcr - 1.5*np.mean(zcr))+1 )

def fundamental_frequency_detection(data, fs):
    f_min = 50
    f_max = 400
    lag_min = int(fs/f_max)
    lag_max = int(fs/f_min)
    index = np.argmax( np.array(
        [autocorrelation_function(data, lag) for lag in range(lag_min, lag_max)] ) )
    return fs/(lag_min+index)