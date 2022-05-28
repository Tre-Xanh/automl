""" Log Mlflow model for using as a coordinator """
from common.mlflow_util import log_model
from common.config import PRJ_DIR
from scorer.coordinator_mlflow import Coordinator

def log_coordinator():
    """ Log Mlflow model for using as a coordinator """
    log_model(
        pre_model=None,
        ml_model=None,
        predictor_model=Coordinator(),
        predictor_code=[
            str(PRJ_DIR / "scorer"),
            str(PRJ_DIR / "common"),
        ],
        conda_env=str(PRJ_DIR / "coordinator_mlflow" / "conda.yml"),
        model_name="MLFLOW_COORDINATOR",
    )
