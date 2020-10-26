import os

import h2o
import joblib
import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
from h2o.frame import H2OFrame
from loguru import logger
from automl.config import TEST_CSV


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
    ML_MODEL = os.getenv("ML_MODEL")
    logger.info(
        f"""Model paths:
    PRE_MODEL {PRE_MODEL}
    ML_MODEL {ML_MODEL}"""
    )
    assert PRE_MODEL and ML_MODEL

    # %% MLflowとH2OAutoMLで保存したモデルをリロードして予測結果を比較
    pred_mlflow = reload_mlflow_predict(dftest)
    logger.info(f"pred_mlflow\n{pred_mlflow}")

    pre_model = joblib.load(PRE_MODEL)
    hf_input = H2OFrame(pre_model.transform(dftest))
    reloaded_h2o = h2o.load_model(ML_MODEL)
    predictions_h2o = reloaded_h2o.predict(hf_input).as_data_frame()["p0"].values
    logger.info(f"predictions_h2o\n{predictions_h2o}")

    assert np.equal(predictions_h2o, pred_mlflow).all()


def test_api():
    dftest = read_dftest()

    data = dftest.to_json(orient="split", index=False)
    res = requests.post(
        "http://127.0.0.1:5000/invocations",
        data=data,
        headers={"Content-type": "application/json"},
    )

    pred_api = np.array(res.json())
    logger.info(pred_api)

    pred_mlflow = reload_mlflow_predict(dftest)
    logger.info(pred_mlflow)
    logger.info(pred_api - pred_mlflow)
    logger.info(np.abs(pred_api - pred_mlflow).max())

    eps = 0.0007
    eps = np.finfo(float).eps
    assert np.abs(pred_api - pred_mlflow).max() <= eps
