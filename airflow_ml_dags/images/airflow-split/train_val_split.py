import os
import pandas as pd
import click

from sklearn.model_selection import train_test_split


@click.command()
@click.option("--input-dir")
@click.option("--output-dir")
def split_data(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    train_data, test_data = train_test_split(data, test_size=.2,
                                             shuffle=True, random_state=19)
    os.makedirs(output_dir, exist_ok=True)
    train_data.to_csv(os.path.join(output_dir, "train_data.csv"))
    test_data.to_csv(os.path.join(output_dir, "val_data.csv"))


if __name__ == "__main__":
    split_data()

