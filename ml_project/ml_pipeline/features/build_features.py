import numpy as np
import pandas as pd
import logging

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


from ml_pipeline.params.feature_params import FeatureParams
from ml_pipeline.params.preprocessing_params import PreprocessingParams


logger = logging.getLogger('__main__.' + __name__)


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan,
                                     strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )

    return categorical_pipeline


class ScalingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

        self.is_fitted = False
        self.mean = None
        self.std = None
        self.upper_percentile = None
        self.lower_percentile = None

    def fit(self, x: pd.DataFrame):
        self.upper_percentile = np.percentile(x, 95, axis=0)
        self.lower_percentile = np.percentile(x, 5, axis=0)
        x_ = x[(x < self.upper_percentile) & (x > self.lower_percentile)]
        self.mean = np.mean(x_)
        self.std = np.std(x_)
        self.is_fitted = True
        return self

    def transform(self, x: pd.DataFrame):
        if not self.is_fitted:
            logger.error("Transformer is not fitted")
            raise NotFittedError()
        else:
            x_ = x.copy()
            x_ = (x_ - self.mean) / self.std
            return x_


def build_numerical_pipeline(preprocessing_params: PreprocessingParams) -> Pipeline:
    processing = [
         ("impute", SimpleImputer(missing_values=np.nan,
                                  strategy="median"))]
    scaler_str = preprocessing_params.scaler

    if scaler_str is not None:
        processing.append(("scaler", ScalingTransformer()))
        logger.info("Custom transformer added to numerical pipeline")

    num_pipeline = Pipeline(processing)

    return num_pipeline


def build_transformer(params: FeatureParams,
                      prcs_params: PreprocessingParams) -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numerical_pipeline",
                build_numerical_pipeline(prcs_params),
                params.numerical_features,
            ),
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),

        ]
    )

    return transformer


def transform_features(transformer: ColumnTransformer, df: pd.DataFrame) -> pd.DataFrame:

    return pd.DataFrame(transformer.transform(df))


def get_target(df: pd.DataFrame, params: FeatureParams) -> pd.Series:
    target = df[params.target_col]

    return target
