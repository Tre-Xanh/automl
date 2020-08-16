import os

import numpy as np
import pandas as pd
import requests


def test_api():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    dftest = pd.read_csv("dftest.csv")
    dftest.drop("Survived", axis="columns", inplace=True)

    data = dftest.to_json(orient="split", index=False)
    res = requests.post(
        "http://127.0.0.1:5000/invocations",
        data=data,
        headers={"Content-type": "application/json"},
    )

    predictions_api = pd.DataFrame(res.json())

    predictions_mlflow = pd.read_csv("predictions_mlflow.csv")
    assert predictions_api.predict.equals(predictions_mlflow.predict)

    assert (predictions_api - predictions_mlflow).abs().max().max() <= np.finfo(
        float
    ).eps
