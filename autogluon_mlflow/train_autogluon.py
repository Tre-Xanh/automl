import mlflow
import pandas as pd
from loguru import logger

from common.config import MAX_TRAIN_SECS, MODEL_DIR, PRJ_DIR, Y_TARGET
from common.mlflow_util import log_model
from common.preprocess import read_processed_data
from scorer.autogluon_mlflow_predictor import AutoGluonPredictor


def train_autogluon():
    mlflow.start_run()
    train, test, pre_model = read_processed_data()
    ml_model = fit_autogluon(train, test)
    log_model(
        pre_model,
        ml_model,
        AutoGluonPredictor(),
        [
            str(PRJ_DIR / "scorer"),
            str(PRJ_DIR / "common"),
        ],
        str(PRJ_DIR / "autogluon_mlflow/conda.yml"),
        model_name="MLFLOW_AUTOGLUON",
    )
    mlflow.end_run()


def optimize_for_deploy(predictor):
    """ https://auto.gluon.ai/tutorials/tabular_prediction/tabular-indepth.html#if-you-encounter-disk-space-issues """
    predictor.save_space()
    predictor.delete_models(models_to_keep="best", dry_run=False)


def fit_autogluon(train: pd.DataFrame, test: pd.DataFrame) -> str:
    from shutil import make_archive

    from autogluon.tabular import TabularPredictor as task

    train = task.Dataset(train)
    logger.debug(train.head())
    ml_model_dir = MODEL_DIR / "autogluon"
    time_limit = MAX_TRAIN_SECS
    metric = "roc_auc"
    predictor = task(label=Y_TARGET, path=ml_model_dir, eval_metric=metric).fit(
        train_data=train, time_limit=time_limit, presets="best_quality",
    )
    predictor.fit_summary()
    val_auc = predictor.info()["best_model_score_val"]

    test = task.Dataset(test)
    logger.debug(test.head())
    test_auc = predictor.evaluate(test)[metric]

    logger.info("val_auc {}, test_auc {}", val_auc, test_auc)

    mlflow.log_param("algos", "AutoGluon")
    mlflow.log_param("max_secs", MAX_TRAIN_SECS)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("test_auc", test_auc)

    optimize_for_deploy(predictor)

    make_archive(ml_model_dir, "zip", root_dir=ml_model_dir)
    ml_model_zip = str(ml_model_dir) + ".zip"
    return ml_model_zip


if __name__ == "__main__":
    train_autogluon()
