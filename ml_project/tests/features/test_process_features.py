import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ml_project.params import FeatureParams, PreprocessingParams
from ml_project.features import (build_transformer,
                                 transform_features,
                                 get_target)
from ml_project.tests.data import fake_data, data


CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET = "target"
SCALER = "StandardScaler"



@pytest.fixture
def feature_params() -> FeatureParams:
    params = FeatureParams(
        categorical_features=CATEGORICAL,
        numerical_features=NUMERICAL,
        target_col=TARGET,
    )
    return params


@pytest.fixture
def preprocessing_params() -> PreprocessingParams:
    params = PreprocessingParams(
        scaler=SCALER
    )
    return params


def test_can_transform_features(fake_data,
                            feature_params):
    for scaler in ["StandardScaler", "MinMaxScaler"]:
        prcs_params = PreprocessingParams(scaler=scaler)
        transformer = build_transformer(feature_params, prcs_params)
        transformer.fit(fake_data)
        features = transform_features(transformer, fake_data)

        numerical_features = features.iloc[:, : len(NUMERICAL)]
        if scaler == "StandardScaler":
            assert np.allclose(numerical_features.mean(axis=0), 0, atol=1e-1), \
                "mean of scaled features not 0"
            assert np.allclose(numerical_features.std(axis=0), 1, atol=1e-1), \
                "std of scaled features not 1"
        elif scaler == "MinMaxScaler":
            assert np.all(abs(numerical_features) <= 1), \
                "scaled features are outside of [-1, 1] range"

        assert features.isna().sum().sum() == 0, \
            "NaNs are present after transform"
        assert features.shape[0] == fake_data.shape[0], \
            "features must contain the same number of rows as original data"
        assert features.shape[1] > fake_data.shape[1], \
            "categorical features were not one hot encoded"


def test_can_get_target(fake_data, feature_params):
    target = get_target(fake_data, feature_params)
    target_vals = set(np.unique(target))

    assert target_vals == {0, 1}, \
        f"target variable contains unexpected values {target_vals}"
