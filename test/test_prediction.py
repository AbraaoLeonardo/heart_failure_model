import pytest
from pathlib import Path
import os
import sys
from pandas import DataFrame

PACKAGE_ROOT = Path(os.path.abspath(os.path.dirname(__file__))).parent
sys.path.append(str(PACKAGE_ROOT))

from prediction_model.config import config
from prediction_model.predict import generate_predictions

@pytest.fixture
def single_prediction():
    data = [[72,1,110,0,25,0,237000,1,140,0,0,65]]
    result = generate_predictions(data)
    return result

def test_single_predi_not_none(single_prediction):
    assert single_prediction is not None

def test_single_pred_is_str(single_prediction):
    print(single_prediction.get('prediction'))
    assert isinstance(single_prediction.get('prediction'),str)


def test_single_pred_validate(single_prediction):
    assert single_prediction.get('prediction') == "Y"
