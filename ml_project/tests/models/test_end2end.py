import os

from ml_project.train_pipeline import train_pipeline
from ml_project.params import read_training_pipeline_params

CONFIG_PATH = "ml_project/tests/models/test_config.yaml"


def test_train_e2e():
    params = read_training_pipeline_params(CONFIG_PATH)
    model_path, metrics = train_pipeline(params)
    assert {"roc auc", "f1", "recall"} == metrics.keys()
    assert metrics["roc auc"] > .5
    assert os.path.exists(model_path)
    assert os.path.exists(params.metric_path)

