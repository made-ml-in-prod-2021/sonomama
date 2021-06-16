import os
import pandas as pd
import click
import json
import pickle

from sklearn.metrics import r2_score



@click.command()
@click.option("--input-dir")
@click.option("--model-dir")
def validate(input_dir: str, model_dir: str):
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)

    validation_data_path = os.path.join(input_dir, 'val_data.csv')
    val_data = pd.read_csv(validation_data_path, index_col=0)
    X_test= val_data.drop('target', axis=1)
    y_test = val_data['target'].values

    score = r2_score(y_test, model.predict(X_test))
    with open(os.path.join(model_dir, 'val_score.json'), 'w') as fout:
        json.dump({"r2_score": score}, fout)


if __name__ == "__main__":
    validate()
