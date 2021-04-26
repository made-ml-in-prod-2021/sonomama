from collections import defaultdict, OrderedDict
from typing import List
import numpy as np
import random
import pandas as pd
import pytest
from faker import Faker

from ml_project.data.make_dataset import read_data, split_train_test_data
from ml_project.params import SplittingParams


DATASET_PATH = "ml_project/data/raw/heart.csv"
TARGET_COL = "target"
FAKE_DATA_SIZE = 300


@pytest.fixture()
def data():
    return read_data(DATASET_PATH)


def add_nans(df: pd.DataFrame, cols: List[str], prob: float = 0.05) -> pd.DataFrame:
    df_ = df.copy()
    for c in cols:
        df_[c] = df[c].apply(
            lambda x: np.nan if np.random.rand() < prob else x)
    return df_


@pytest.fixture()
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
    assert df_fake.isna().sum().sum() > 0, \
        "fake data does not have NaNs"

    return df_fake


def test_load_dataset(fake_data):
    assert len(fake_data) == 300
    assert TARGET_COL in fake_data.keys(), \
        "target columns is missing from the dataframe"


def test_split_dataset(fake_data):
    test_size = 0.2
    splitting_params = SplittingParams(random_state=1, test_size=test_size,)
    train, test = split_train_test_data(fake_data, splitting_params)
    assert train.shape[0] == FAKE_DATA_SIZE * (1 - test_size)
    assert test.shape[0] == FAKE_DATA_SIZE * test_size
