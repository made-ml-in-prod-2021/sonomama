import os
import pandas as pd
import click
import pickle


@click.command()
@click.option("--input-dir")
@click.option("--model-dir")
@click.option("--output-dir")
def predict(input_dir: str, model_dir: str, output_dir: str):
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as fin:
        model = pickle.load(fin)

    data = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=0)

    os.makedirs(output_dir, exist_ok=True)
    prediction = pd.DataFrame({'target': model.predict(data)})
    prediction.to_csv(os.path.join(output_dir, 'predictions.csv'))


if __name__ == "__main__":
    predict()
