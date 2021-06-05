import os
import numpy as np
import pandas as pd
import click

from sklearn.impute import SimpleImputer


@click.command()
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=0)
    target = pd.read_csv(os.path.join(input_dir, "target.csv"), index_col=0)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_imp = pd.DataFrame(imp.fit_transform(data), columns=data.columns)

    data = pd.concat((data_imp, target), axis=1)
    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)


if __name__ == '__main__':
    preprocess()
