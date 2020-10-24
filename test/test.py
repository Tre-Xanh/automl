import os

import h2o
import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
from h2o.frame import H2OFrame

from config import TEST_CSV, logger
from model import Preproc


def test_load_model():
    df_input = pd.read_csv(TEST_CSV)

    MLFLOW_MODEL = os.getenv("MLFLOW_MODEL")
    PRE_MODEL = os.getenv("PRE_MODEL")
    H2O_MODEL = os.getenv("H2O_MODEL")
    logger.info(
        f"""Model paths:
    PRE_MODEL {PRE_MODEL}
    MLFLOW_MODEL {MLFLOW_MODEL}
    H2O_MODEL {H2O_MODEL}"""
    )
    assert PRE_MODEL and MLFLOW_MODEL and H2O_MODEL

    # %% MLflowとH2OAutoMLで保存したモデルをリロードして予測結果を比較
    reloaded_mlflow = mlflow.pyfunc.load_model(MLFLOW_MODEL)
    predictions_mlflow: pd.DataFrame = reloaded_mlflow.predict(df_input)
    logger.info(f"predictions_mlflow\n{predictions_mlflow}")

    pre_model = joblib.load(PRE_MODEL)
    hf_input = H2OFrame(pre_model.transform(df_input))
    reloaded_h2o = h2o.load_model(H2O_MODEL)
    predictions_h2o = reloaded_h2o.predict(hf_input).as_data_frame()
    logger.info(f"predictions_h2o\n{predictions_h2o}")

    assert predictions_h2o.equals(predictions_mlflow)


def test_api():
    dftest = pd.read_csv(TEST_CSV)
    dftest.drop("Survived", axis="columns", inplace=True)

    data = dftest.to_json(orient="split", index=False)
    res = requests.post(
        "http://127.0.0.1:5000/invocations",
        data=data,
        headers={"Content-type": "application/json"},
    )

    pred_api = pd.DataFrame(res.json())

    pred_mlflow = pd.read_csv("data/predictions_mlflow.csv")
    assert pred_api.predict.equals(pred_mlflow.predict)

    assert ((pred_api - pred_mlflow).abs() <= np.finfo(float).eps).all(axis=None)
