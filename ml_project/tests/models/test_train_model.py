import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
import pickle
from py._path.local import LocalPath

from ml_pipeline.params import TrainingParams
from ml_pipeline.models.utils import (
    train_model,
    predict,
    save_model
)


def test_can_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())

    assert isinstance(model, RandomForestClassifier), \
        f"unexpected model {model.__dict__['base_estimator']}"
    predicted_shape = predict(model, features).shape[0]
    assert predicted_shape == target.shape[0], \
        f"predicted shape {predicted_shape} while should be {target.shape[0]}"


def test_can_save_model(tmpdir: LocalPath):
    model = RandomForestClassifier(n_estimators=50)
    expected_path = tmpdir.join("model.pkl")
    save_model(model, expected_path)
    with open(expected_path, "rb") as f:
        model = pickle.load(f)

    assert isinstance(model, RandomForestClassifier), \
        f"unexpected model {model.__dict__['base_estimator']}"




