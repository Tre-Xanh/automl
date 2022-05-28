import os
import asyncio
import httpx
import mlflow
import pandas as pd
from loguru import logger
from common.mlflow_api import async_request_api

class Coordinator(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        logger.info("Start ...")
        # logger.add("/app/log/coordinator_mlflow_{time}.log", rotation="00:00")
        self.predictor_A = os.getenv("PREDICTOR_A_URI")
        self.predictor_B = os.getenv("PREDICTOR_B_URI")
        logger.debug(f"PREDICTOR_A_URI {self.predictor_A}")
        logger.debug(f"PREDICTOR_B_URI {self.predictor_B}")
        assert self.predictor_A
        assert self.predictor_B
        logger.info("... Done")

    async def async_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Start ... ")
        async with httpx.AsyncClient() as client:
            tasks = [async_request_api(df=df, client=client, scoring_uri=uri) for uri in [self.predictor_A, self.predictor_B]]
            pred_A_df, pred_B_df = await asyncio.gather(*tasks)
        preds = pd.concat([
            pred_A_df.rename(columns={"proba": "pred_A"}),
            pred_B_df.rename(columns={"proba": "pred_B"}),
        ], axis=1)
        logger.debug(f"Predictions\n{preds}")
        logger.info("... Done")
        return preds

    def predict(self, context, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Start ...")
        logger.debug(f"df\n{df}")
        preds = asyncio.run(self.async_predict(df))
        logger.debug(f"Predictions\n{preds}")
        logger.info("... Done")
        return preds
