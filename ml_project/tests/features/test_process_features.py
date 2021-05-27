import numpy as np
from ml_pipeline.features import (ScalingTransformer,
                                  build_transformer,
                                  transform_features,
                                  get_target)


CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL = ["age", "trestbps", "chol", "thalach", "oldpeak"]
TARGET = "target"
SCALER = "true"


def test_can_scale_features(fake_data):
    transformer = ScalingTransformer()
    num_data = fake_data[NUMERICAL].dropna()
    transformer.fit(num_data)
    transformed_num_features = transformer.transform(num_data)

    assert transformed_num_features.shape[1] == len(NUMERICAL), \
        f"shape of scaled features is {transformed_num_features.shape[1]}, " \
        f"expected {len(NUMERICAL)}"

    assert np.allclose(transformed_num_features.mean(), 0, atol=1), \
        f"mean of scaled features not 0: {transformed_num_features.mean()}"
    assert np.allclose(transformed_num_features.std(), 1, atol=1), \
        f"std of scaled features is not 1: {transformed_num_features.std()}"

    upper_percentile = np.percentile(num_data, 95, axis=0)
    lower_percentile = np.percentile(num_data, 5, axis=0)
    clipped_num_features = num_data[(num_data < upper_percentile) & (num_data > lower_percentile)]
    expected_num_features = (num_data - np.mean(clipped_num_features)) / np.std(clipped_num_features)

    assert transformed_num_features.equals(expected_num_features), \
        "scaled features are different from expected"


def test_can_transform_features(fake_data,
                                feature_params,
                                preprocessing_params):

    transformer = build_transformer(feature_params, preprocessing_params)
    transformer.fit(fake_data)
    features = transform_features(transformer, fake_data)

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
