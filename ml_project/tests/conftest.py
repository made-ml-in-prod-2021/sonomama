from collections import defaultdict, OrderedDict
from typing import List
import numpy as np
import pandas as pd
import pytest
from faker import Faker

from ml_pipeline.params import FeatureParams, PreprocessingParams
from ml_pipeline.data.make_dataset import read_data
from ml_pipeline.features import (build_transformer,
                                  transform_features,
                                  get_target)
from tests.data import DATASET_PATH, FAKE_DATA_SIZE
from tests.features import CATEGORICAL, NUMERICAL, SCALER, TARGET


@pytest.fixture(scope="session")
def data():
    return read_data(DATASET_PATH)


def add_nans(df: pd.DataFrame, cols: List[str], prob: float = 0.05) -> pd.DataFrame:
    df_ = df.copy()
    for c in cols:
        df_[c] = df[c].apply(
            lambda x: np.nan if np.random.rand() < prob else x)
    return df_


@pytest.fixture(scope="session")
def fake_data(data, size=FAKE_DATA_SIZE):
    faker = Faker()
    fake_data = defaultdict(list)
    cp_vals = data["cp"].value_counts()
    restecg_vals = data["restecg"].value_counts()
    oldpeak_vals = data["oldpeak"].value_counts()
    slope_vals = data["slope"].value_counts()
    ca_vals = data["ca"].value_counts()
    thal_vals = data["thal"].value_counts()

    data_size = data.shape[0]
    for i in range(size):
        fake_data["age"].append(faker.random_int(30, 80))
        fake_data["sex"].append(faker.random_int(0, 1))
        fake_data["cp"].append(faker.random_element(
            elements=OrderedDict(
                [(cp_vals.index[i], cp_vals.iloc[i] / data_size)
                 for i in range(cp_vals.shape[0])]))
                              )
        fake_data["trestbps"].append(faker.random_int(
            data["trestbps"].min(),
            data["trestbps"].max()))

        fake_data["chol"].append(
            faker.random_int(data["chol"].min(), data["chol"].max()))
        fake_data["fbs"].append(faker.random_int(0, 1)),
        fake_data["restecg"].append(
            faker.random_element(elements=OrderedDict([
                (restecg_vals.index[i], restecg_vals.iloc[i] / data_size)
                for i in range(restecg_vals.shape[0])]))
        )
        fake_data["thalach"].append(
            faker.random_int(data["thalach"].min(),
                             data["thalach"].max()))
        fake_data["exang"].append(faker.random_int(0, 1))
        fake_data["oldpeak"].append(
            faker.random_element(elements=OrderedDict([
                (oldpeak_vals.index[i], oldpeak_vals.iloc[i] / data_size)
                for i in range(oldpeak_vals.shape[0])]))
        )

        fake_data["slope"].append(
            faker.random_element(elements=OrderedDict([
                (slope_vals.index[i], slope_vals.iloc[i] / data_size)
                for i in range(slope_vals.shape[0])]))
        )

        fake_data["ca"].append(faker.random_element(
            elements=OrderedDict([
                (ca_vals.index[i], ca_vals.iloc[i] / data_size)
                for i in range(ca_vals.shape[0])]))
        )
        fake_data["thal"].append(faker.random_element(
            elements=OrderedDict([
                (thal_vals.index[i], thal_vals.iloc[i] / data_size)
                for i in range(thal_vals.shape[0])]))
        )
        fake_data["target"].append(faker.random_int(0, 1))

    df_fake = pd.DataFrame(fake_data)
    df_fake = add_nans(df_fake, cols=["age", "thal", "oldpeak"])
    df_fake.to_csv("tests/data/fake.csv")

    return df_fake


@pytest.fixture(scope="session")
def feature_params() -> FeatureParams:
    params = FeatureParams(
        categorical_features=CATEGORICAL,
        numerical_features=NUMERICAL,
        target_col=TARGET,
    )
    return params


@pytest.fixture(scope="session")
def preprocessing_params() -> PreprocessingParams:
    params = PreprocessingParams(
        scaler=SCALER
    )
    return params


@pytest.fixture(scope="session")
def features_and_target(fake_data,
                        feature_params,
                        preprocessing_params):
    transformer = build_transformer(feature_params,
                                    preprocessing_params)
    transformer.fit(fake_data)
    features = transform_features(transformer, fake_data)
    target = get_target(fake_data, feature_params)

    return features, target
