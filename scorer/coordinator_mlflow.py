import os
import mlflow
import pandas as pd
from loguru import logger
from common.mlflow_api import request_api

class Coordinator(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.predictor_A = os.getenv("PREDICTOR_A_URI")
        self.predictor_B = os.getenv("PREDICTOR_B_URI")
        logger.info(f"PREDICTOR_A_URI {self.predictor_A}")
        logger.info(f"PREDICTOR_B_URI {self.predictor_B}")
        assert self.predictor_A
        assert self.predictor_B

    def predict(self, context, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Starting ... ")
        logger.debug(f"PREDICTOR_A_URI {self.predictor_A}")
        pred_A_df = request_api(df, self.predictor_A)
        logger.debug(f"PREDICTOR_B_URI {self.predictor_B}")
        pred_B_df = request_api(df, self.predictor_B)
        preds = pd.concat([
            pred_A_df.rename(columns={"proba": "pred_A"}),
            pred_B_df.rename(columns={"proba": "pred_B"}),
        ], axis=1)
        logger.info(f"Predictions\n{preds}")
        return preds
