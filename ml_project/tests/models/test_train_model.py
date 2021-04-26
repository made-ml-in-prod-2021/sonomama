import pytest
import pandas as pd
from typing import Tuple
from sklearn.ensemble import RandomForestClassifier
import pickle
from py._path.local import LocalPath

from ml_project.tests.data import fake_data, data
from ml_project.tests.features import feature_params, preprocessing_params
from ml_project.features import (build_transformer,
                                 transform_features,
                                 get_target)
from ml_project.params import TrainingParams

from ml_project.models.utils import (
    train_model,
    predict,
    save_model
)


@pytest.fixture
def features_and_target(fake_data,
                        feature_params,
                        preprocessing_params):
    transformer = build_transformer(feature_params,
                                    preprocessing_params)
    transformer.fit(fake_data)
    features = transform_features(transformer, fake_data)
    target = get_target(fake_data, feature_params)
    return features, target


def test_train_model(features_and_target: Tuple[pd.DataFrame, pd.Series]):
    features, target = features_and_target
    model = train_model(features, target, train_params=TrainingParams())
    assert isinstance(model, RandomForestClassifier)
    assert predict(model, features).shape[0] == target.shape[0]


def test_save_model(tmpdir: LocalPath):
    model = RandomForestClassifier(n_estimators=50)
    expected_path = tmpdir.join("model.pkl")
    real_path = save_model(model, expected_path)
    assert real_path == expected_path
    with open(real_path, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestClassifier)




