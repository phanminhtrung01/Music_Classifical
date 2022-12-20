import tensorflow as tf
import numpy as np
import training as tn
import os

TRAIN_PATH = "training/my_model.h5"
DATABASE_TEST = "test"
DATABASE_TEST_ = "./test/data.json"

if os.path.exists(DATABASE_TEST_):
    os.remove(DATABASE_TEST_)

tn.save_mfcc(DATABASE_TEST, DATABASE_TEST_, n_mfcc=32)

X, y = tn.load_data(DATABASE_TEST_)
M = X[..., np.newaxis]
model = tf.keras.models.load_model(TRAIN_PATH)
# Testing the model on never seen before data.
tn.predict(model, M[2], y[0])
