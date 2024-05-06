import os
import joblib
import pandas as pd

from prediction_model.config import config

def data_load():
    data_path = os.path.join(config.DATAPATH,config.CSV_FILE)
    _dataframe = pd.read_csv(data_path)
    return _dataframe

def save_pipeline(pipeline_to_save):
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    joblib.dump(pipeline_to_save, save_path)
    print(f"Pipeline saved with success as {config.MODEL_NAME}")

def load_model():
    save_path = os.path.join(config.SAVE_MODEL_PATH, config.MODEL_NAME)
    model = joblib.load(save_path)
    print("model has been loaded")
    return model
