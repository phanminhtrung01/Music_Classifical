import numpy as np
import matplotlib.pyplot as plt
import librosa
from librosa import display as librosadisplay
from scipy.io import wavfile
from pydub import AudioSegment
import sys
import math
import logging
import music21
import statistics
import tensorflow as tf
from base64 import b64decode
from IPython.display import Audio, Javascript

uploaded_file_name = r'D:\IdeaProjects\Music Classifical\song\1.wav'

MAX_ABS_INT16 = 32768.0
EXPECTED_SAMPLE_RATE = 16000
SAMPLE_RATE = 44100


def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file)
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file


converted_audio_file = convert_audio_for_model(uploaded_file_name)

sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')


def plot_stft(x, sample_rate, show_black_and_white=False):
    x_stft = np.abs(librosa.stft(x, n_fft=2048))
    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    x_stft_db = librosa.amplitude_to_db(x_stft, ref=np.max)
    if show_black_and_white:
        librosadisplay.specshow(data=x_stft_db, y_axis='log',
                                sr=sample_rate, cmap='gray_r')
    else:
        librosadisplay.specshow(data=x_stft_db, y_axis='log', sr=sample_rate)

    plt.colorbar(format='%+2.0f dB')
    plt.show()


