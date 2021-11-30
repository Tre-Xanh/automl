# %%
import tempfile
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import h2o

import joblib
import mlflow
import mlflow.h2o
import mlflow.pyfunc
import pandas as pd
from h2o import H2OFrame
from h2o.automl import H2OAutoML
from h2o.frame import H2OFrame
from loguru import logger

from automl.config import (
    MAX_TRAIN_SECS,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    TMP_DIR,
    Y_TARGET,
)


def h2o_fit(train: H2OFrame, test: H2OFrame) -> str:
    x = train.columns
    y = Y_TARGET
    x.remove(y)
    logger.info(f"x={x}, y={y}")

    train[y] = train[y].asfactor()
    test[y] = test[y].asfactor()

    aml = H2OAutoML(max_runtime_secs=MAX_TRAIN_SECS)
    aml.train(x=x, y=y, training_frame=train)

    # View the AutoML Leaderboard
    logger.info(aml.leaderboard)

    val_auc = aml.leader.model_performance(xval=True).auc()
    test_auc = aml.leader.model_performance(test).auc()

    logger.info("val_auc {}, test_auc {}", val_auc, test_auc)
    mlflow.log_param("algos", "H2OAutoML")
    mlflow.log_param("max_secs", MAX_TRAIN_SECS)
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
        self.pre_model: Preproc = joblib.load(context.artifacts["pre_model"])
        self.ml_model = h2o.load_model(context.artifacts["ml_model"])

    def predict(self, context, df_input: pd.DataFrame) -> pd.DataFrame:
        hf_input = H2OFrame(self.pre_model.transform(df_input))
        output: H2OFrame = self.ml_model.predict(hf_input)
        proba = output.as_data_frame()["p0"].values
        return pd.DataFrame({"proba": proba})


def read_processed_data():
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    pre_model = str(MODEL_DIR / "prep.model")
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
    ml_model = h2o_fit(H2OFrame(train), H2OFrame(test))

    log_model(pre_model, ml_model, H2OPredictor())


def log_model(pre_model, ml_model, python_model):
    # %% MLflowで学習済みの前処理・モデルを保存
    conda_env = "./conda.yml"
    code_path = ["automl", "test"]  # 重要：ローカル Python ソースコードの所在地
    artifacts = dict(
        pre_model=pre_model,
        #
        ml_model=ml_model,
    )

    artifact_path = "pyfunc"
    logger.debug(f"mlflow.pyfunc.log_model {artifacts} ... ")
    mlflow_model_info = dict(
        artifact_path=artifact_path,
        python_model=python_model,
        code_path=code_path,
        artifacts=artifacts,
        conda_env=conda_env,
    )
    mlflow.pyfunc.log_model(**mlflow_model_info)
    logger.info(f"mlflow log_model {mlflow_model_info}")
    mlflow_model = mlflow.get_artifact_uri(artifact_path)
    prefix = "file://"
    if mlflow_model.startswith(prefix):
        mlflow_model = mlflow_model[len(prefix) :]

    mlflow.end_run()

    (TMP_DIR / "run_env.sh").write_text(
        f"""
set -a
MLFLOW_MODEL={mlflow_model}
PYTHONPATH={mlflow_model}/code
TESTPATH={mlflow_model}/code/test
"""
    )


def train_autogluon():
    mlflow.start_run()
    train, test, pre_model = read_processed_data()
    ml_model = fit_autogluon(train, test)
    log_model(pre_model, ml_model, AutoGluonPredictor())
    mlflow.end_run()


class AutoGluonPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from autogluon.tabular import TabularPredictor

        logger.info(f"artifacts {context.artifacts}")
        self.pre_model: Preproc = joblib.load(context.artifacts["pre_model"])
        self.ml_model = TabularPredictor.load(context.artifacts["ml_model"])

    def predict(self, context, input: pd.DataFrame) -> pd.DataFrame:
        input = self.pre_model.transform(input)
        proba = self.ml_model.predict_proba(input)
        return pd.DataFrame({"proba": proba})


def fit_autogluon(train: pd.DataFrame, test: pd.DataFrame) -> str:
    from autogluon.tabular import TabularPredictor, TabularDataset

    train = TabularDataset(train)
    logger.debug(train.head())
    model_path = tempfile.mkdtemp(dir=MODEL_DIR)
    predictor = TabularPredictor(
        label=Y_TARGET,
        path=model_path,
    ).fit(
        train_data=train,
        time_limit=MAX_TRAIN_SECS,
    )
    predictor.fit_summary()
    val_auc = predictor.info()["best_model_score_val"]

    test = TabularDataset(test)
    logger.debug(test.head())
    test_eval = predictor.evaluate(test)
    logger.info("test_eval {}", test_eval)

    test_auc = test_eval["roc_auc"]
    logger.info("val_auc {}, test_auc {}", val_auc, test_auc)

    mlflow.log_param("algos", "AutoGluon")
    mlflow.log_param("max_secs", MAX_TRAIN_SECS)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("test_auc", test_auc)

    return model_path
