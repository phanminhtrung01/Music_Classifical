import os
import shutil

import keras.models
import numpy as np
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

from training import training as tn

JSON_PATH_TEST = "./test/music/"
MODEL_PATH = "./model/my_model1.h5"

app = FastAPI()


class SongClassification(BaseModel):
    id_song: str
    file_name: str
    predict: str
    accuracy: str


@app.get('/')
def getting():
    return 'Hello'


@app.get('/get/songs')
def getAllSong():
    return 'All'


@app.post('/predict/song/{song_id}')
async def upload(
        files: UploadFile,
        song_id: int,
):
    test_predict = []
    path_song = JSON_PATH_TEST + "1"

    os.mkdir(path_song)
    path_json = os.path.join(path_song, 'data.json')
    path_song = os.path.join(path_song, files.filename)
    with open(path_song, 'wb') as f:
        f.write(files.file.read())
        f.close()
        model = keras.models.load_model(MODEL_PATH)

        tn.save_mfcc_test(JSON_PATH_TEST, path_json)
        x = tn.load_data_test(path_json)
        shutil.rmtree(os.path.dirname(path_json))
        x = x[..., np.newaxis]

        for i in range(0, x.shape[0]):
            predict = tn.predict(model=model, X=x[i])
            test_predict.append(predict)

        test = np.array(test_predict)
        test = test.flatten()
        test1 = set(test)
        x = 0
        for i in test1:
            if test_predict.count(i) > x:
                x = test_predict.count(i)
                y = i

    return {"songId": song_id, "predict": int(y)}
