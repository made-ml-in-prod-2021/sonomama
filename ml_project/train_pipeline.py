import json
import logging
import logging.config
import yaml

import click

from ml_project.data import read_data, split_train_test_data

from ml_project.params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)

from ml_project.features import (
    build_transformer,
    transform_features,
    get_target
)

from ml_project.models import (
    train_model,
    predict,
    evaluate_model,
    save_model,
)


logger = logging.getLogger(__name__)


def setup_logging(training_pipeline_params: TrainingPipelineParams):
    with open(training_pipeline_params.logging_config_path) as config_fin:
        config = yaml.safe_load(config_fin)
        logging.config.dictConfig(config)


def train_pipeline(training_pipeline_params: TrainingPipelineParams):
    logger.info(f"start train pipeline with params {training_pipeline_params}\n")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")
    train_df, test_df = split_train_test_data(
        data, training_pipeline_params.splitting_params
    )
    logger.info(f"train_df.shape is {train_df.shape}")
    logger.info(f"test_df.shape is {test_df.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params,
                                    training_pipeline_params.preprocessing_params)
    transformer.fit(train_df)
    train_features = transform_features(transformer, train_df)
    logger.debug(f"train_features.shape is {train_features.shape}")
    train_target = get_target(train_df,
                              training_pipeline_params.feature_params)

    model = train_model(
        train_features, train_target, training_pipeline_params.train_params
    )
    logger.info(f"{training_pipeline_params.train_params.model_type} trained")

    test_features = transform_features(transformer, test_df)
    test_target = get_target(test_df, training_pipeline_params.feature_params)

    logger.info(f"test_features.shape is {test_features.shape}")
    predicted = predict(
        model,
        test_features
    )

    metrics = evaluate_model(
        predicted,
        test_target
    )

    logger.info(f"metrics on test: {metrics}")
    with open(training_pipeline_params.metric_path, "w") as metric_file:
        json.dump(metrics, metric_file)
        logger.info(f"metrics dumped")

    path_to_model = save_model(model,
                               training_pipeline_params.output_model_path)

    return path_to_model, metrics


@click.command()
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    params = read_training_pipeline_params(config_path)
    setup_logging(params)
    train_pipeline(params)


if __name__ == "__main__":
    train_pipeline_command()
