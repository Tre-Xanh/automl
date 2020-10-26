import tempfile
from pathlib import Path

import joblib
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from automl.config import DATA_URI, PROCESSED_DATA_DIR, TEST_CSV


class Preproc:
    SCALE_COLS = ["Age", "Fare"]

    def __init__(self) -> None:
        pass

    def fit(self, df) -> None:
        logger.info(f"Original Columns {df.columns}")
        logger.info(f"MinMaxScaler fit {self.SCALE_COLS}")
        self.scaler = MinMaxScaler().fit(df[self.SCALE_COLS])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop("Name", axis="columns")
        logger.debug(f"MinMaxScaler transform {self.SCALE_COLS}")
        df[self.SCALE_COLS] = self.scaler.transform(df[self.SCALE_COLS])
        logger.debug(f"df head\n{df.head()}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save_model(self, model_path=None) -> str:
        model_path = model_path or os.path.join(tempfile.mkdtemp(), "prep.model")
        joblib.dump(self, model_path)
        return model_path


def preprocess(csv=DATA_URI):
    # %% 実験データのダウンロード
    data = pd.read_csv(csv)
    logger.debug("Downloaded data")

    # %% 学習・テストデータの分割
    dftrain: pd.DataFrame
    dftest: pd.DataFrame
    dftrain, dftest = train_test_split(data)
    logger.info(f"train_test_split {data.shape} -> {dftrain.shape} + {dftest.shape}")

    # Save test dataframe for later tests
    Path("../data").mkdir(exist_ok=True)
    dftest.to_csv(TEST_CSV, index=False)

    # %% データ前処理
    prep = Preproc()
    train = prep.fit_transform(dftrain)
    pre_model = prep.save_model(PROCESSED_DATA_DIR / "prep.model")
    test = prep.transform(dftest)
    train.to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    return train, test, pre_model
