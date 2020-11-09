import mlflow
import pandas as pd
from loguru import logger

from automl.config import MODEL_DIR, PRJ_DIR, PROCESSED_DATA_DIR


def read_processed_data():
    train = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")
    pre_model = str(MODEL_DIR / "prep.model")
    return train, test, pre_model


def log_model(pre_model, ml_model, python_model):
    # %% MLflowで学習済みの前処理・モデルを保存
    conda_env = "./conda.yml"
    artifacts = dict(
        pre_model=pre_model,
        #
        ml_model=ml_model,
    )

    artifact_path = "automl"
    logger.debug(f"mlflow.pyfunc.log_model {artifacts} ... ")
    mlflow_model_info = dict(
        artifact_path=artifact_path,
        python_model=python_model,
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

    (PRJ_DIR / ".trained.env").write_text(
        f"""
MLFLOW_MODEL={mlflow_model}
""".strip()
    )
