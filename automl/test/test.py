import os

import h2o
import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
from h2o.frame import H2OFrame

from automl.config import TEST_CSV, logger


def read_dftest():
    dftest = pd.read_csv(TEST_CSV)
    dftest.drop("Survived", axis="columns", inplace=True)
    return dftest


def reload_mlflow_predict(dftest):
    MLFLOW_MODEL = os.getenv("MLFLOW_MODEL")
    logger.info(f"MLFLOW_MODEL {MLFLOW_MODEL}")
    reloaded_mlflow = mlflow.pyfunc.load_model(MLFLOW_MODEL)
    pred_mlflow: pd.DataFrame = reloaded_mlflow.predict(dftest)
    return pred_mlflow


def test_load_model():
    dftest = read_dftest()

    PRE_MODEL = os.getenv("PRE_MODEL")
    H2O_MODEL = os.getenv("H2O_MODEL")
    logger.info(
        f"""Model paths:
    PRE_MODEL {PRE_MODEL}
    H2O_MODEL {H2O_MODEL}"""
    )
    assert PRE_MODEL and H2O_MODEL

    # %% MLflowとH2OAutoMLで保存したモデルをリロードして予測結果を比較
    pred_mlflow = reload_mlflow_predict(dftest)
    logger.info(f"pred_mlflow\n{pred_mlflow}")

    pre_model = joblib.load(PRE_MODEL)
    hf_input = H2OFrame(pre_model.transform(dftest))
    reloaded_h2o = h2o.load_model(H2O_MODEL)
    predictions_h2o = reloaded_h2o.predict(hf_input).as_data_frame()
    logger.info(f"predictions_h2o\n{predictions_h2o}")

    assert predictions_h2o.equals(pred_mlflow)


def test_api():
    dftest = read_dftest()

    data = dftest.to_json(orient="split", index=False)
    res = requests.post(
        "http://127.0.0.1:5000/invocations",
        data=data,
        headers={"Content-type": "application/json"},
    )

    pred_api = pd.DataFrame(res.json())

    pred_mlflow = reload_mlflow_predict(dftest)
    assert pred_api.predict.equals(pred_mlflow.predict)

    assert ((pred_api - pred_mlflow).abs() <= np.finfo(float).eps).all(axis=None)
