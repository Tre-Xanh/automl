from __future__ import annotations
import logging
import h2o
from h2o import H2OFrame
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import tempfile
import joblib

logger = logging.getLogger(__name__)
logger.debug("h2o_mlflow")


class Preproc:
    SCALE_COLS = ["Age", "Fare"]

    def __init__(self) -> None:
        h2o.init()

    def fit(self, df) -> None:
        logger.info(f"Original Columns {df.columns}")
        self.scaler = MinMaxScaler().fit(df[self.SCALE_COLS])

    def transform(self, df: pd.DataFrame) -> H2OFrame:
        df = df.drop("Name", axis="columns")
        df[self.SCALE_COLS] = self.scaler.transform(df[self.SCALE_COLS])
        logger.info(f"df info\n{df.describe()}")
        return H2OFrame(df)

    def fit_transform(self, df) -> H2OFrame:
        self.fit(df)
        return self.transform(df)

    def save_model(self) -> str:
        model_path = os.path.join(tempfile.mkdtemp(), "prep.model")
        joblib.dump(self, model_path)
        return model_path

    @classmethod
    def load_model(cls, model_path) -> Preproc:
        prep: Preproc = joblib.load(model_path)
        return prep
