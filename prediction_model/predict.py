import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os
import sys

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.processing.data_handling import load_model

classification_pipeline = load_model()

def generate_predictions(data_input):
    columns =     columns = ["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure","platelets","serum_creatinine","serum_sodium","sex","smoking","time"]
    
    data = pd.DataFrame(data=data_input,columns=columns)
    pred = classification_pipeline.predict(data[config.FEATURES])[0]
    output = str(np.where(pred==1,"Y","N"))
    result = {"prediction": output}
    return result


if __name__ == "__main__":
    generate_predictions()