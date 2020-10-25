# %%
import os
import tempfile
from pathlib import Path

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

from automl.config import (
    DATA_URI,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    TEST_CSV,
    TMP_DIR,
    Y_TARGET,
    logger,
    max_mins,
)


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
        logger.debug(f"df info\n{df.describe()}")
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    def save_model(self, model_path=None) -> str:
        model_path = model_path or os.path.join(tempfile.mkdtemp(), "prep.model")
        joblib.dump(self, model_path)
        return model_path


def h2o_fit(train: H2OFrame, test: H2OFrame) -> str:
    x = train.columns
    y = Y_TARGET
    x.remove(y)
    logger.info(f"x={x}, y={y}")

    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    aml = H2OAutoML(max_runtime_secs=max_mins * 60)
    aml.train(x=x, y=y, training_frame=train)

    # View the AutoML Leaderboard
    logger.info(aml.leaderboard)

    val_auc = aml.leader.model_performance(xval=True).auc()
    test_auc = aml.leader.model_performance(test).auc()

    logger.info("val_auc {}, test_auc {}", val_auc, test_auc)
    mlflow.log_param("algos", "H2OAutoML")
    mlflow.log_param("max_mins", max_mins)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("test_auc", test_auc)

    # mlflow.h2o.log_model(aml.leader, "model")

    model_path = tempfile.mkdtemp(dir=MODEL_DIR)
    model_path = h2o.save_model(model=aml.leader, path=model_path)
    logger.debug(f"h2o.save_model to {model_path} DONE")
    return model_path


class H2OPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):

        h2o.init()
        logger.info(f"artifacts {context.artifacts}")
        self.h2o_model = h2o.load_model(context.artifacts["h2o_model"])
        self.pre_model: Preproc = joblib.load(context.artifacts["pre_model"])

    def predict(self, context, df_input: DataFrame):
        # Convert input from Pandas
        hf_input = H2OFrame(self.pre_model.transform(df_input))
        output: H2OFrame = self.h2o_model.predict(hf_input)

        # Convert output to Pandas
        return output.as_data_frame()


def preprocess(csv=DATA_URI):
    # %% 実験データのダウンロード
    data = pd.read_csv(csv)
    logger.debug("Downloaded data")

    # %% 学習・テストデータの分割
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


def read_processed_data():
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    pre_model = str(PROCESSED_DATA_DIR / "prep.model")
    return train, test, pre_model


def train_h2o():
    """A pipeline to
    - Read CSV and preprocess data
    - Train using H2OAutoML
    - Save trained models
    """
    mlflow.start_run()
    train, test, pre_model = read_processed_data()

    # %% 機械学習 AutoML
    h2o.init()
    h2o_model = h2o_fit(H2OFrame(train), H2OFrame(test))

    # %% MLflowで学習済みの前処理・モデルを保存
    conda_env = "./conda.yml"
    code_path = ["automl"]  # 重要：ローカル Python ソースコードの所在地
    artifacts = dict(
        pre_model=pre_model,
        #
        h2o_model=h2o_model,
    )

    artifact_path = "pyfunc"
    logger.debug(f"mlflow.pyfunc.log_model {artifacts} ... ")
    mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=H2OPredictor(),
        code_path=code_path,
        artifacts=artifacts,
        conda_env=conda_env,
    )
    logger.debug(f"mlflow.pyfunc.log_model ... DONE")
    mlflow_model = mlflow.get_artifact_uri(artifact_path)
    prefix = "file://"
    if mlflow_model.startswith(prefix):
        mlflow_model = mlflow_model[len(prefix) :]

    mlflow.end_run()

    pre_model = Path(mlflow_model) / "artifacts" / os.path.basename(pre_model)
    h2o_model = Path(mlflow_model) / "artifacts" / os.path.basename(h2o_model)
    (TMP_DIR / "run_env.sh").write_text(
        f"""
set -a
MLFLOW_MODEL={mlflow_model}
PRE_MODEL={pre_model}
H2O_MODEL={h2o_model}
PYTHONPATH={mlflow_model}/code
TESTPATH={mlflow_model}/code/automl/test
"""
    )


def train_autogluon():
    from autogluon.tabular import TabularPrediction as task

    mlflow.start_run()
    train, test, pre_model = read_processed_data()

    train = task.Dataset(train)
    logger.debug(train.head())
    dir = tempfile.mkdtemp()
    time_limits = max_mins * 60
    metric = "roc_auc"
    predictor = task.fit(
        train_data=train,
        label=Y_TARGET,
        output_directory=dir,
        eval_metric=metric,
        time_limits=time_limits,
        presets="best_quality",
    )
    predictor.fit_summary()
    val_auc = predictor.info()["best_model_score_val"]

    test = task.Dataset(test)
    logger.debug(test.head())
    test_auc = predictor.evaluate(test)

    logger.info("val_auc {}, test_auc {}", val_auc, test_auc)

    mlflow.log_param("algos", "AutoGluon")
    mlflow.log_param("max_mins", max_mins)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("test_auc", test_auc)

    mlflow.end_run()

    return dir


if __name__ == "__main__":
    train_autogluon()
