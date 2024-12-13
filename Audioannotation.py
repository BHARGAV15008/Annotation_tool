import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import pandas as pd
import librosa
from python_speech_features import logfbank


def CWT_parameter(segment, scales):
    """
    Computes the CWT for the current segment.

    Args:
        segment (np.array): The current segment of audio file.
        scales (np.array): Scales for CWT.

    Returns:
        2D numpy.array of complex datatype: Extracted CWT for the particular segment
    """
    
    # Compute the CWT for the current segment
    cwt_matrix = signal.cwt(segment, signal.morlet, scales)

    return cwt_matrix


def bark_filter_audio(sr,signal):
    """
    Computes the CWT for the current segment.

    Args:
        sr (int): The sampling rate of our audio file.
        signal (np.array): Signal is the current segment of the audio file.

    Returns:
        2D numpy.array of float datatype: Extracted bark features of the given segment
    """
    
    # Increase FFT size
    NFFT = len(signal)

    # Compute bark features for current segment.
    bark_features = logfbank(signal, samplerate=sr, nfilt=26,nfft=NFFT)
    return bark_features


def MFCC_parameters(sampling_rate, segment):
    """
    Computes the MFCC for the current segment.

    Args:
        sampling_rate (int): The sampling rate of our audio file.
        segment (np.array): The current segment of the audio file.

    Returns:
        2D numpy.array of float datatype: Extracted bark features of the given segment
    """
    
    # Compute mfcc features for current segment.
    mfcc = librosa.feature.mfcc(y=segment, sr=sampling_rate)

    return mfcc


def STFT_parameters(segment):
    """
    Computes the STFT for the current segment.

    Args:
        segment (np.array): The current segment of the audio file.

    Returns:
        2D numpy.array of complex datatype: Extracted bark features of the given segment
    """

    # Compute STFT features for the current segment
    stft = librosa.stft(y=segment)

    return stft

# Main function
def main(audio_path):
    
    """
    Takes audio path, finds sampling rate and audio signal, reduces its dimension if its more than 1.
    Defines the parameters for the CWT. Defines scale, scaling factor, sampling interval.
    Calulates frequencies corresponding to the scales.
    Defines segment length and compute the segment lenght and number of segments.
    Creates lists to store audio features, loops through each segment of segment duration, finds each of the four features and appends the features of each segment to their respective lists.
    Creates dataframe to store all the features witht their correspoding segment duration in a CSV file.

    Args:
        audio_path (str): The path of an audio file.

    Returns:
        df(pandas.Dataframe): The created dataframe containing the audio features and segment duration
    """

    # Load the WAV file
    sampling_rate, audio_signal = wavfile.read(audio_path)

    # Convert audio signal to mono if necessary (if stereo)
    if audio_signal.ndim > 1:
        audio_signal = np.mean(audio_signal, axis=1)

    # Define the parameters for the CWT
    scales = np.arange(1, 51)
    
    # Define segment length and compute the segment lenght and number of segments
    segment_duration = 1  # Segment duration in seconds
    segment_length = int(sampling_rate * segment_duration)
    num_segments = len(audio_signal) // segment_length

    # List to store feature parameters
    mfcc_params_list = []
    stft_params_list = []
    cwt_params_list = []
    bark_params_list = []

    # Create segment for 1 second
    for i in range(0, len(audio_signal), segment_length):
        segment = audio_signal[i:i + segment_length]

        # Compute the MFCC parameters for each segment
        mfcc_params = MFCC_parameters(sampling_rate, segment)
        mfcc_params_list.append(mfcc_params.flatten())

        # Compute the STFT parameters for each segment
        stft_params = STFT_parameters(segment)
        stft_params_list.append(stft_params.flatten())

        # compute the bark filter for each segment
        bark_params = bark_filter_audio(sampling_rate, segment)
        bark_params_list.append(bark_params.flatten())

        # Compute the CWT parameters for each segment
        cwt_params = CWT_parameter(segment, scales)
        cwt_params_list.append(cwt_params.flatten())

    # Create a pandas DataFrame to store the feature parameters
    df = pd.DataFrame({
        "Second": range(1, num_segments+2),
        "MFCC Parameters": mfcc_params_list,
        "STFT_Parameters": stft_params_list,
        "CWT Parameters": cwt_params_list,
        "Bark Parameters": bark_params_list
    })
    
    return df
