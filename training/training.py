import json
import math
import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras

SAMPLE_RATE = 22050
DURATION = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION
MODEL_PATH1 = "./model/my_model1.h5"
MODEL_PATH2 = "./model/my_model2.h5"
MODEL_PATH3 = "./model/my_model3.h5"
MODEL_PATH4 = "./model/my_model4.h5"
MODEL_PATH5 = "./model/my_model5.h5"


def save_mfcc(dataset_path, json_path, n_mfcc=32, n_fft=2048,
              hop_length=512, num_segments=5):
    # Data storage dictionary
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": [],
    }
    samples_ps = int(SAMPLES_PER_TRACK / num_segments)  # ps = per segment
    expected_vects_ps = math.ceil(samples_ps / hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensuring not at root
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_comp = dirpath.split("\\")
            semantic_label = dirpath_comp[-1]
            data["mapping"].append(semantic_label)
            print(f"Processing: {semantic_label}")

            # process files for specific genre
            for f in filenames:
                if f == str("001.00054.wav"):
                    # As librosa only read files <1Mb
                    continue
                else:
                    file_path = os.path.join(dirpath, f)
                    length = librosa.get_duration(filename=file_path)
                    duration = 30
                    offset = 0

                    num_run = 0
                    while offset <= length - duration:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, offset=offset, duration=duration)
                        offset += duration
                        # load audio file
                        for s in range(num_segments):
                            start_sample = samples_ps * s
                            finish_sample = start_sample + samples_ps

                            mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                        sr=sr, n_fft=n_fft, n_mfcc=n_mfcc,
                                                        hop_length=hop_length)

                            mfcc = mfcc.T

                            # store mfcc if it has expected length
                            if len(mfcc) == expected_vects_ps:
                                data["mfcc"].append(mfcc.tolist())
                                data["labels"].append(i - 1)
                                num_run = num_run + 1
                                print(f"{file_path}, segment: {num_run}")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def save_mfcc_test(dataset_path, json_path, n_mfcc=32, n_fft=2048,
                   hop_length=512, num_segments=5):
    data = {
        "mfcc": [],
    }
    samples_ps = int(SAMPLES_PER_TRACK / num_segments)  # ps = per segment
    expected_vects_ps = math.ceil(samples_ps / hop_length)

    # loop through all the genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # ensuring not at root
        if dirpath is not dataset_path:
            # save the semantic label
            dirpath_comp = dirpath.split("\\")
            semantic_label = dirpath_comp[-1]
            print(f"Processing: {semantic_label}")

            # process files for specific genre
            for f in filenames:
                if f == str("001.00054.wav"):
                    # As librosa only read files <1Mb
                    continue
                else:
                    file_path = os.path.join(dirpath, f)
                    length = librosa.get_duration(filename=file_path)
                    duration = 30
                    offset = 0

                    num_run = 0
                    while offset <= length - duration:
                        signal, sr = librosa.load(file_path, sr=SAMPLE_RATE, offset=offset, duration=duration)
                        offset = offset + duration
                        # load audio file
                        for s in range(num_segments):
                            start_sample = samples_ps * s
                            finish_sample = start_sample + samples_ps

                            mfcc = librosa.feature.mfcc(y=signal[start_sample:finish_sample],
                                                        sr=sr, n_fft=n_fft, n_mfcc=n_mfcc,
                                                        hop_length=hop_length)

                            mfcc = mfcc.T

                            # store mfcc if it has expected length
                            if len(mfcc) == expected_vects_ps:
                                data["mfcc"].append(mfcc.tolist())
                                num_run = num_run + 1
                                print(f"{file_path}, segment: {num_run + 1}")

    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)


def load_data(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    X = np.array(data["mfcc"])
    y = np.array(data["labels"])

    return X, y


def load_data_test(data_path):
    with open(data_path, "r") as f:
        data = json.load(f)
    X = np.array(data["mfcc"])

    return X


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

    X_train = X_train[..., np.newaxis]
    X_val = X_val[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_alexnet(input_shape, num_classes):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    model.add(tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    # model = keras.models.Sequential([
    #     keras.layers.Input(shape=input_shape),
    #     keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dropout(0.4),
    #
    #     keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dropout(0.4),
    #
    #     keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dropout(0.4),
    #
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(64, activation='relu'),
    #     keras.layers.BatchNormalization(),
    #     keras.layers.Dropout(0.4),
    #     keras.layers.Dense(32, activation='relu'),
    #     keras.layers.Dense(units=num_classes, activation='softmax'),
    #
    # ])

    return model


def predict(model, X):
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print(f"Predicted index: {predicted_index}")
    return predicted_index


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


def training(dataset_path, json_path, model_train_path):
    # save_mfcc(dataset_path, json_path, n_mfcc=32)

    if os.path.exists(model_train_path):
        model_train_path = MODEL_PATH1

    if os.path.exists(model_train_path):
        model_train_path = MODEL_PATH2

    if os.path.exists(model_train_path):
        model_train_path = MODEL_PATH3

    if os.path.exists(model_train_path):
        model_train_path = MODEL_PATH4

    if os.path.exists(model_train_path):
        model_train_path = MODEL_PATH5

    name_model = model_train_path.split("/")[-1]

    X, y = load_data(json_path)
    X_train, X_val, X_test, y_train, y_val, y_test = split_in_sets_cnn(X, y,
                                                                       train_size=0.8,
                                                                       val_split=0.8)
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print(input_shape)

    model = build_alexnet(input_shape, 10)

    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    # model.compile(
    #     loss='sparse_categorical_crossentropy',
    #     optimizer=tf.keras.optimizers.RMSprop(),
    #     metrics=['accuracy'],
    # )
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    model.summary()

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50, batch_size=64)

    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_accuracy}")

    predicted_index = predict(model, X_test[10])

    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=64)
    print("The test loss is ", test_loss)
    print("The best accuracy is: ", test_acc * 100)

    # cm = confusion_matrix(y_test, predicted_index)
    # print(cm)

    if test_accuracy >= 0.8:
        model.save(model_train_path)

    plot_history(history, name_model)
