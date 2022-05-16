from typing import List

import mlflow
from loguru import logger

from common.config import PRJ_DIR


def log_model(
    pre_model,
    ml_model,
    predictor_model,
    predictor_code: List[str],
    conda_env: str,
    model_name="MLFLOW_MODEL",
):
    # %% MLflowで学習済みの前処理・モデルを保存
    artifacts = {}
    if pre_model:
        artifacts["pre_model"] =pre_model
    if ml_model:
        artifacts["ml_model"] =ml_model
    artifact_path = "automl"
    logger.debug(artifacts)
    logger.debug(conda_env)
    mlflow_model_info = dict(
        artifact_path=artifact_path,
        python_model=predictor_model,
        artifacts=artifacts,
        conda_env=conda_env,
        code_path=predictor_code,
    )
    mlflow.pyfunc.log_model(**mlflow_model_info)
    logger.info(f"mlflow log_model {mlflow_model_info}")
    mlflow_model = mlflow.get_artifact_uri(artifact_path)
    prefix = "file://"
    if mlflow_model.startswith(prefix):
        mlflow_model = mlflow_model[len(prefix) :]

    mlflow.end_run()

    with (PRJ_DIR / ".trained.env").open("a") as env_fp:
        env_fp.write(f"""{model_name}={mlflow_model}\n""")
