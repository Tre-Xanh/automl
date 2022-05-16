""" Log Mlflow model for using as a gateway """
from common.mlflow_util import log_model
from common.config import PRJ_DIR
from scorer.mlflow_gateway import PredictorGateway

def log_gateway():
    """ Log Mlflow model for using as a gateway """
    log_model(
        pre_model=None,
        ml_model=None,
        predictor_model=PredictorGateway(),
        predictor_code=[
            str(PRJ_DIR / "scorer"),
            str(PRJ_DIR / "common"),
        ],
        conda_env=str(PRJ_DIR / "mlflow_gateway" / "conda.yml"),
        model_name="MLFLOW_GATEWAY",
    )
