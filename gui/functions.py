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
    return (1 / len(data)) * (np.sum(np.power(data, 2)))


def volume(data):
    return np.sqrt(short_time_energy(data))


def zero_crossing_rate(data):
    n = len(data)
    return (1 / (2 * n)) * (np.sum(np.abs(np.sign(data[1:n]) - np.sign(data[0:n - 1]))))


def autocorrelation_function(data, l):
    n = len(data)
    return np.sum(np.multiply((data[l:n]), (data[0:n - l])))


def average_magnitude_difference(data, l):
    n = len(data)
    return np.sum(np.abs(np.sign(data[l:n]) - np.sign(data[0:n - l])))


def scale_data(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scale = max((-np.min(data)), np.max(data))
    return np.divide(data, scale)
