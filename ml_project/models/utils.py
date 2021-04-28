import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score, f1_score, recall_score

from ml_project.params.train_params import TrainingParams

SklearnClassifier = Union[LogisticRegression, RandomForestClassifier]


def train_model(
        features: pd.DataFrame,
        target: pd.Series,
        train_params: TrainingParams) -> SklearnClassifier:

    if train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            penalty="l1", solver="liblinear"
        )
    elif train_params.model_type == "RandomForestClassifier":
        model = RandomForestClassifier(max_depth=train_params.max_depth,
                                       random_state=train_params.random_state)
    else:
        raise NotImplementedError()

    model.fit(features, target)

    return model


def predict(
        model: SklearnClassifier,
        features: pd.DataFrame) -> np.ndarray:
    predicted = model.predict(features)

    return predicted


def evaluate_model(
        predicted: np.ndarray,
        target: pd.Series) -> Dict[str, float]:

    return {
        "roc auc": roc_auc_score(target, predicted),
        "f1": f1_score(target, predicted),
        "recall": recall_score(target, predicted),
    }


def save_model(model: SklearnClassifier,
               output: str):
    with open(output, "wb") as f:
        pickle.dump(model, f)

