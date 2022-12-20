import os
import stempeg
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow import keras
import soundfile as sf
import numpy as np
from sklearn.model_selection import train_test_split

SAMPLE_RATE = 44100
TRAIN_PATH = "D:/AI/musdb18/train/"
MIX_PATH = "song/data/mixtures/"
MODEL_TRAIN_PATH = "./model/my_model_song.h5"


def split_in_sets_cnn(X, y, train_size, val_split):
    print(f"X total = {X.shape}")
    print(f"y total = {y.shape}")
    X_train, X_ev, y_train, y_ev = train_test_split(X, y, train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_ev, y_ev, train_size=val_split)

    print(f"X train = {X_train.shape}")
    print(f"y train = {y_train.shape}")
    print(f"X val = {X_val.shape}")
    print(f"y val = {y_val.shape}")
    print(f"X test = {X_test.shape}")
    print(f"y test = {y_test.shape}")

    return X_train, X_val, X_test, y_train, y_val, y_test


def extract_vocal(fname):
    stems, _ = stempeg.read_stems(fname, stem_id=[0, 4, 3])
    stems = stems.astype(np.float32)

    master = stems[0, :, :]
    vocal = stems[1, :, :]
    other = stems[2, :, :]

    return master, vocal, other


def create_a_song(PATH):
    master, vocal, other = extract_vocal(PATH)
    sf.write('1.wav', vocal, SAMPLE_RATE)


def create_data():
    all_songs = os.listdir(TRAIN_PATH)
    # Generate mixtures, others and vocals files
    for i, fname in enumerate(all_songs):
        if i == 1:
            break
        master, vocal, other = extract_vocal(os.path.join(TRAIN_PATH, fname))
        sf.write('data/mixtures/{}.wav'.format(i), master, SAMPLE_RATE)
        sf.write('data/vocal/{}.wav'.format(i), vocal, SAMPLE_RATE)
        sf.write('data/others/{}.wav'.format(i), other, SAMPLE_RATE)


def split_sets():
    # Skip 0.5 seconds
    skip = SAMPLE_RATE // 2
    # Mixtures part as training data
    X_part = []
    # Vocal part as label
    Y_part = []

    # Define win_len
    length = 255
    hop_size = 1024
    win_len = hop_size * length

    for mixture_path in tqdm(MIX_PATH):
        vocal_path = mixture_path.replace('mixtures', 'vocals')
        # Load x and y song
        x, sr = librosa.load(mixture_path, sr=SAMPLE_RATE)
        y, sr = librosa.load(vocal_path, sr=SAMPLE_RATE)

        # Padding win_len 0 for x and y
        x_pad = np.pad(x, (0, win_len), mode="constant")
        y_pad = np.pad(y, (0, win_len), mode="constant")

        l = len(x_pad)

        for i in range(0, l - win_len - skip, skip):
            x_part = x_pad[i:i + win_len]
            y_part = y_pad[i:i + win_len]

            X_part.append(x_part)
            Y_part.append(y_part)

    return X_part, Y_part


def build_alexnet(input_shape, num_classes):
    model = keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(),

        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.BatchNormalization(),

        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation='softmax'),
    ])

    return model


def plot_history(hist, name_model):
    plt.figure(figsize=(20, 15))
    fig, axs = plt.subplots(2)
    # accuracy subplot
    axs[0].plot(hist.history["accuracy"], label="train accuracy")
    axs[0].plot(hist.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval " + name_model)

    # Error subplot
    axs[1].plot(hist.history["loss"], label="train error")
    axs[1].plot(hist.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


create_a_song(r"D:\AI\musdb18\test\Al James - Schoolboy Facination.stem.mp4")
# create_data()
# X_part, Y_part = split_sets()
#
# X_train, X_val, Y_train, Y_val = train_test_split(X_part, Y_part, test_size=0.1, random_state=42)
# print(X_train)
# input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
# model = build_alexnet(input_shape, 1)
# optimizer = keras.optimizers.Adam(learning_rate=0.0001)
# model.compile(optimizer=optimizer,
#               loss="sparse_categorical_crossentropy",
#               metrics=["accuracy"])
#
# model.summary()
#
# history = model.fit(X_train, Y_train,
#                     validation_data=(X_val, Y_val),
#                     epochs=100, batch_size=64)
#
# test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
# print(f"Test accuracy: {test_accuracy}")
#
# if test_accuracy >= 0.8:
#     model.save(MODEL_TRAIN_PATH)
#
# plot_history(history, 1)
#
#
# model.evaluate()
