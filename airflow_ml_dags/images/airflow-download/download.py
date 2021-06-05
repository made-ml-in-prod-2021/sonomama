import os

import click
from sklearn.datasets import load_diabetes


@click.command()
@click.argument("output_dir")
def download(output_dir: str):
    x, y = load_diabetes(return_X_y=True, as_frame=True)
    os.makedirs(output_dir, exist_ok=True)
    x.to_csv(os.path.join(output_dir, "data.csv"))
    y.to_csv(os.path.join(output_dir, "target.csv"))


if __name__ == '__main__':
    download()
