from training import training as tn

DATASET_PATH = "./dataset/genres"
DATASET_PATH1 = "./dataset/genres1"
# JSON_PATH = "./dataset/data/data.json"
JSON_PATH_REALITY = "./dataset/data/data_reality.json"
JSON_PATH_REALITY1 = "./dataset/data/data_reality1.json"
JSON_PATH_TEST = "test/music/song1/data_test.json"
MODEL_PATH = "./model/my_model.h5"
MODEL_PATH1 = "./model/my_model1.h5"
MODEL_PATH2 = "./model/my_model2.h5"
TEST_PATH = "./test/music/001"

if __name__ == "__main__":
    tn.training(DATASET_PATH, JSON_PATH_REALITY, MODEL_PATH)

    # model = keras.models.load_model(MODEL_PATH)
    # model1 = keras.models.load_model(MODEL_PATH1)
    # model2 = keras.models.load_model(MODEL_PATH2)
    #
    # # tn.save_mfcc_test(TEST_PATH, JSON_PATH_TEST)
    # X = tn.load_data_test(JSON_PATH_TEST)
    # X = X[..., np.newaxis]
    #
    # for i in range(0, X.shape[0]):
    #     # print("==================0===================")
    #     # tn.predict(model=model, X=X[i], y=0)
    #     # print("==================1===================")
    #     # tn.predict(model=model1, X=X[i], y=0)
    #     print("==================2===================")
    #     tn.predict(model=model2, X=X[i], y=0)
