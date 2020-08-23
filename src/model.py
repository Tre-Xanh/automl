from __future__ import annotations

import os
import tempfile

import h2o
import joblib
import mlflow
import mlflow.h2o
import mlflow.pyfunc
import pandas as pd
from h2o import H2OFrame
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from config import DATA_URI, logger

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


class Learner:
    def __init__(self) -> None:
        import h2o

        h2o.init()

    def fit(self, train: H2OFrame, test: H2OFrame) -> str:

        x = train.columns
        y = "Survived"
        x.remove(y)
        logger.info(f"x={x}, y={y}")

        train[y] = train[y].asfactor()
        test[y] = test[y].asfactor()

        max_mins = 1
        aml = H2OAutoML(max_runtime_secs=max_mins * 60)
        aml.train(x=x, y=y, training_frame=train)

        # View the AutoML Leaderboard
        logger.info(aml.leaderboard)

        perf = aml.leader.model_performance(test)
        test_auc, test_aucpr = perf.auc(), perf.aucpr()

        logger.info(f"test_auc {test_auc}, test_aucpr {test_aucpr}")
        mlflow.log_param("max_mins", max_mins)
        mlflow.log_metric("test_auc", test_auc)
        mlflow.log_metric("test_aucpr", test_aucpr)

        mlflow.h2o.log_model(aml.leader, "model")

        import h2o

        model_path = h2o.save_model(
            model=aml.leader, path=tempfile.mkdtemp(), force=True
        )
        return model_path


class Predictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):

        h2o.init()
        logger.info(f"artifacts {context.artifacts}")
        self.h2o_model: Learner = h2o.load_model(context.artifacts["h2o_model"])
        self.pre_model: Preproc = Preproc.load_model(context.artifacts["pre_model"])

    def predict(self, context, df_input: DataFrame):
        # Convert input from Pandas
        hf_input = self.pre_model.transform(df_input)
        output: H2OFrame = self.h2o_model.predict(hf_input)

        # Convert output to Pandas
        return output.as_data_frame()


def train():
    mlflow.start_run()

    # %% 実験データのダウンロード
    data = pd.read_csv(DATA_URI)
    logger.debug("Downloaded data")

    # %% 学習・テストデータの分割
    dftrain: pd.DataFrame
    dftest: pd.DataFrame
    dftrain, dftest = train_test_split(data)
    logger.info(f"train_test_split {data.shape} -> {dftrain.shape} + {dftest.shape}")

    # %% データ前処理
    prep = Preproc()
    train: H2OFrame = prep.fit_transform(dftrain)
    pre_model = prep.save_model()
    test: H2OFrame = prep.transform(dftest)

    # %% 機械学習 AutoML
    h2o_model = Learner().fit(train, test)

    # %% MLflowで学習済みの前処理・モデルを保存
    conda_env = "./conda.yaml"
    code_path = ["."]  # 重要：ローカル Python ソースコードの所在地
    artifacts = dict(
        pre_model=pre_model,
        #
        h2o_model=h2o_model,
    )
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=Predictor(),
        code_path=code_path,
        artifacts=artifacts,
        conda_env=conda_env,
    )

    mlflow_model = os.path.join(tempfile.mkdtemp(), "main.model")
    mlflow.pyfunc.save_model(
        path=mlflow_model,
        python_model=Predictor(),
        code_path=code_path,
        artifacts=artifacts,
        conda_env=conda_env,
    )

    mlflow.end_run()

    # Save test dataframe for later tests
    from pathlib import Path

    Path("../data").mkdir(exist_ok=True)
    dftest.to_csv("../data/dftest.csv", index=False)

    # Log final model path
    logger.info(
        f"""### Test and run the saved model as ###

export PRE_MODEL={pre_model}
export H2O_MODEL={h2o_model}
export MLFLOW_MODEL={mlflow_model}
export PYTHONPATH=src

pytest test/test_load_model.py

mlflow models serve -m $MLFLOW_MODEL
"""
    )
