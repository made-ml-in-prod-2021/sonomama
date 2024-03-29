from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml_pipeline.params import SplittingParams


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def split_train_test_data(
    data: pd.DataFrame, params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_data, test_data = train_test_split(
        data, test_size=params.test_size, random_state=params.random_state
    )
    return train_data, test_data

