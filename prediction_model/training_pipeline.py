from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing import data_handling
from prediction_model.pipeline import classification_pipeline

def training_performace():
    train_data = data_handling.data_load()
    train_y = train_data[config.TARGET]
    classification_pipeline.fit(train_data[config.FEATURES], train_y)
    data_handling.save_pipeline(classification_pipeline)

if __name__ == "__main__":
    training_performace()