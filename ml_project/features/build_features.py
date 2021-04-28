import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler


from ml_project.params.feature_params import FeatureParams
from ml_project.params.preprocessing_params import PreprocessingParams


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan,
                                     strategy="most_frequent")),
            ("ohe", OneHotEncoder()),
        ]
    )

    return categorical_pipeline


def build_numerical_pipeline(preprocessing_params: PreprocessingParams) -> Pipeline:
    processing = [
         ("impute", SimpleImputer(missing_values=np.nan,
                                  strategy="median"))]
    scaler_str = preprocessing_params.scaler

    if scaler_str is not None:
        if scaler_str == "StandardScaler":
            processing.append(("scaler", StandardScaler()))
        elif scaler_str == "MinMaxScaler":
            processing.append(("scaler", MinMaxScaler()))

        elif scaler_str == "RobustScaler":
            processing.append(("scaler", RobustScaler()))

        else:
            raise NotImplementedError()

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
