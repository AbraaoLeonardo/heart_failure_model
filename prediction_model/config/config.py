import os
import pathlib
import prediction_model

PACKAGE_ROOT = pathlib.Path(prediction_model.__file__).resolve().parent
DATAPATH = os.path.join(PACKAGE_ROOT,"data")

CSV_FILE = "heart_failure_clinical_records.csv"

MODEL_NAME = 'classfication.pkl'
SAVE_MODEL_PATH = os.path.join(PACKAGE_ROOT,"trained_models")

TARGET = "DEATH_EVENT"

FEATURES = ["age",
"anaemia","creatinine_phosphokinase",
"diabetes",
"ejection_fraction",
"high_blood_pressure",
"platelets",
"serum_creatinine",
"serum_sodium",
"sex",
"smoking",
"time"
]
RANDOM_SEED = 5
N_ESTIMATORS=200
MAX_DEPTH = 100
CRITERION = 'gini'
MAX_LEAF_NODES = 50
CV = 5
N_JOBS=1,
SCORING='accuracy',
VERBOSE=0