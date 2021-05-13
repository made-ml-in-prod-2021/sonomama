import pandas as pd
import json
import requests

PATH_TO_DATA = "data/requests.csv"


if __name__ == "__main__":
    data_df = pd.read_csv(PATH_TO_DATA)
    data_df.index = data_df["Id"]
    data_df.drop(columns=["Id"], inplace=True)
    try:
        response = requests.post("http://0.0.0.0:8000/predict",
                                 json.dumps(data_df.to_dict("records")))
    except requests.exceptions.RequestException as err:
        raise SystemExit(err)
    response_json = response.json()
    for i, id_ in enumerate(data_df.index):
        print(f"{id_}: {response_json[i]['target']}")
