import librosa
import numpy as np
from IPython.lib.display import Audio
from matplotlib import pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
from librosa import display as librosadisplay


EXPECTED_SAMPLE_RATE1 = 44100
uploaded_file_name = r"D:\AI\music_vn\vocal\mp3\Gap Me Trong Mo - Thuy Chi [vocals] (mp3cut.net) (2).mp3"


def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file)
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE1).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file


converted_audio_file = convert_audio_for_model(uploaded_file_name)

sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

# Show some basic information about the audio.
duration = len(audio_samples) / sample_rate
print(f'Sample rate: {sample_rate} Hz')
print(f'Total duration: {duration:.2f}s')
print(f'Size of the input: {len(audio_samples)}')

# Let's listen to the wav file.
Audio(audio_samples, rate=sample_rate)

_ = plt.plot(audio_samples)

MAX_ABS_INT16 = 32768.0


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


plot_stft(audio_samples / MAX_ABS_INT16, sample_rate=EXPECTED_SAMPLE_RATE1)
plt.show()
