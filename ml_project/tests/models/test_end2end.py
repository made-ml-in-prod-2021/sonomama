import os

from ml_pipeline.train_pipeline import train_pipeline
from ml_pipeline.params import read_training_pipeline_params

CONFIG_PATH = "ml_project/tests/models/test_config.yaml"


def test_can_train_e2e():
    params = read_training_pipeline_params(CONFIG_PATH)
    metrics = train_pipeline(params)

    assert {"roc auc", "f1", "recall"} == metrics.keys(), \
        f"unexpected metrics output {metrics.keys()}"
    assert metrics["roc auc"] > .5, \
        f"ROC AUC score {metrics['roc auc']} < 0.5"
    assert os.path.exists(params.output_model_path), \
        f"no such file {params.output_model_path}"
    assert os.path.exists(params.metric_path), \
        f"no such file {params.metric_path}"

