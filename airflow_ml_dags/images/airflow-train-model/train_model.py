import os
import pandas as pd
import click
import json
import pickle

from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold, GridSearchCV


@click.command()
@click.option("--input-dir")
@click.option("--output-dir")
def train_model(input_dir: str, output_dir: str):
    train_data = pd.read_csv(os.path.join(input_dir, "train_data.csv"), index_col=0)
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target'].values
    model = Lasso()
    params = {'alpha': [5e-4, 1e-3, 5e-3]}
    cv = KFold(n_splits=5, shuffle=True, random_state=19)
    grid = GridSearchCV(model, params, cv=cv, n_jobs=-1, scoring="r2")
    grid.fit(X_train, y_train)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'model.pkl'), 'wb') as fout:
        pickle.dump(grid.best_estimator_, fout)

    with open(os.path.join(output_dir, 'train_score.json'), 'w') as fout:
        json.dump({"r2_score": grid.best_score_}, fout)


if __name__ == "__main__":
    train_model()


