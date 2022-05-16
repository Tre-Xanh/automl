import os
import warnings

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

import mlflow.pyfunc
import numpy as np
import pandas as pd
import requests
from loguru import logger

from common.config import TEST_CSV, Y_TARGET_
from common.mlflow_api import request_api


def read_dftest():
    dftest = pd.read_csv(TEST_CSV)
    dftest.drop(columns=Y_TARGET_, inplace=True)
    return dftest


def reload_mlflow_predict(dftest):
    MLFLOW_MODEL = os.getenv("MLFLOW_AUTOGLUON")
    logger.info(f"MLFLOW_MODEL {MLFLOW_MODEL}")
    reloaded_mlflow = mlflow.pyfunc.load_model(MLFLOW_MODEL)
    pred_mlflow: pd.DataFrame = reloaded_mlflow.predict(dftest)
    return pred_mlflow


def test_reload_model():
    dftest = read_dftest()
    pred_mlflow = reload_mlflow_predict(dftest)["proba"]
    logger.debug(pred_mlflow)


def test_api():
    scoring_uri = os.getenv("SCORING_URI", "http://127.0.0.1:5000/invocations")
    dftest = read_dftest()
    pred_df = request_api(dftest, scoring_uri)
    pred_api = pred_df.iloc[:, 0]
    pred_mlflow = reload_mlflow_predict(dftest)["proba"]
    preds = pd.DataFrame(dict(pred_api=pred_api, pred_mlflow=pred_mlflow,))
    preds["pred_diff"] = preds.pred_api - preds.pred_mlflow
    print(preds)
    if (preds["pred_diff"] != 0).any():
        logger.warning(
            f"""nonzero_diff
{preds[preds["pred_diff"] != 0]}
in #{preds.shape[0]} samples
"""
        )

    eps = np.finfo(float).eps
    assert preds["pred_diff"].abs().max() <= eps


if __name__ == "__main__":
    # test_reload_model()
    test_api()
