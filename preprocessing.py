import numpy as np
import itertools
from moviepy.editor import VideoFileClip
import scipy.signal as signal
from scipy.io import wavfile
import librosa
from python_speech_features import logfbank
import threading
import os
import shutil

bark_params_list = [[], [], [], [], []]

def fetchVideo(df,video_path):
    """
    Fetches the audio file from the given video and saves it as WAV.

    Args:
        df (DataFrame): DataFrame containing video information.
        video_path (str): Path to the video file.
    """

    # Create directory
    if not os.path.exists("preprocessing_Audio"):
        os.makedirs("preprocessing_Audio")

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        class_name = row["Class"]
        start_frame = row["Start Frame"]
        end_frame = row["End Frame"]
        tracker_ID = row["Tracker ID"]

        #cap = cv2.VideoCapture(video_path)
        clip = VideoFileClip(video_path)

        # Extract the subclip based on start and end frames
        subclip = clip.subclip(start_frame / clip.fps, end_frame / clip.fps)

        # Set the audio of the subclip to the original audio
        audio = subclip.audio

        # Set the output audio filename
        output_path = f"{class_name}_{tracker_ID}.wav"
        # Write the audio to the output file
        audio.write_audiofile("preprocessing_Audio/"+output_path, codec="pcm_s16le")


def feature_extraction(sampling_rate, audio_signal, l):
    """
    Extracts audio features using logfbank.

    Args:
        sampling_rate (int): Sampling rate of the audio.
        audio_signal (ndarray): Audio signal.
        l (int): Index for storing the features.
    """
    bark_params = logfbank(audio_signal, samplerate=sampling_rate, nfilt=26,nfft=len(audio_signal))
    bark_params_list[l].append(bark_params.flatten())


def audioAnnotation(df):
    """
    Performs audio annotation by extracting various audio features.

    Args:
        df (DataFrame): DataFrame containing audio information.

    Returns:
        DataFrame: DataFrame with added audio features.
    """
    
    global bark_params_list
    # List to store feature parameters
    mfcc_params_list = []
    stft_params_list = []
    cwt_params_list = []


    for i in range(len(df['Class'])):
        # Load the WAV file
        wav_file = f'preprocessing_Audio/{df["Class"][i]}_{df["Tracker ID"][i]}.wav'
        sampling_rate, audio_signal = wavfile.read(wav_file)

        # Convert audio signal to mono if necessary (if stereo)
        if audio_signal.ndim > 1:
            audio_signal = np.mean(audio_signal, axis=1)

        # Define the parameters for the CWT
        scales = np.arange(1, 51)

        aslength = int(len(audio_signal)/5)

        cwt_matrix = signal.cwt(audio_signal, signal.morlet, scales)
        mfcc = librosa.feature.mfcc(y=audio_signal, sr=sampling_rate)
        stft = librosa.stft(y=audio_signal)

        mfcc_params_list.append(mfcc)
        cwt_params_list.append(cwt_matrix)
        stft_params_list.append(stft)

        threads1 = threading.Thread(target=feature_extraction, args=(sampling_rate, audio_signal[0:aslength], 0))
        threads2 = threading.Thread(target=feature_extraction, args=(sampling_rate, audio_signal[aslength:2*aslength], 1))
        threads3 = threading.Thread(target=feature_extraction, args=(sampling_rate, audio_signal[2*aslength:3*aslength], 2))
        threads4 = threading.Thread(target=feature_extraction, args=(sampling_rate, audio_signal[3*aslength:4*aslength], 3))
        threads5 = threading.Thread(target=feature_extraction, args=(sampling_rate, audio_signal[4*aslength:len(audio_signal)], 4))

        threads1.start()
        threads2.start()
        threads3.start()
        threads4.start()
        threads5.start()

        threads1.join()
        threads2.join()
        threads3.join()
        threads4.join()
        threads5.join()

    bark_params_list = [list(itertools.chain(*sublist)) for sublist in zip(*bark_params_list)]
    print(np.array(bark_params_list).shape)

    df["MFCC FEATURE"] = mfcc_params_list
    df["STFT FEATURE"] = stft_params_list
    df["CWT FEATURE"] = cwt_params_list
    df["BARK FEATURE"] = bark_params_list

    # Clean the directory and delete it
    shutil.rmtree("preprocessing_Audio")
    return df