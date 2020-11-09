import joblib
import mlflow
import pandas as pd
from loguru import logger

from automl.common import log_model, read_processed_data
from automl.config import MAX_TRAIN_SECS, MODEL_DIR, Y_TARGET
from automl.preprocess import Preproc


def train_autogluon():
    mlflow.start_run()
    train, test, pre_model = read_processed_data()
    ml_model = fit_autogluon(train, test)
    log_model(pre_model, ml_model, AutoGluonPredictor())
    mlflow.end_run()

def unpack_model_zip(ml_model_zip: str) -> str:
    import tempfile
    from shutil import unpack_archive
    ml_model_dir = tempfile.mkdtemp()
    unpack_archive(ml_model_zip, extract_dir=ml_model_dir)
    return ml_model_dir

class AutoGluonPredictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):

        from autogluon.tabular import TabularPrediction as task

        logger.info(f"artifacts {context.artifacts}")
        self.pre_model: Preproc = joblib.load(context.artifacts["pre_model"])

        ml_model = unpack_model_zip(context.artifacts["ml_model"])
        self.ml_model = task.load(ml_model)

        # Keeping models in memory. BUG: loading deleted models
        # self.ml_model.persist_models()

    def predict(self, context, input: pd.DataFrame) -> pd.DataFrame:
        input = self.pre_model.transform(input)
        proba = self.ml_model.predict_proba(input)
        return pd.DataFrame({"proba": proba})


def optimize_for_deploy(predictor):
    """ https://auto.gluon.ai/tutorials/tabular_prediction/tabular-indepth.html#if-you-encounter-disk-space-issues """
    predictor.save_space()
    predictor.delete_models(models_to_keep="best", dry_run=False)


def fit_autogluon(train: pd.DataFrame, test: pd.DataFrame) -> str:
    from shutil import make_archive

    from autogluon.tabular import TabularPrediction as task

    train = task.Dataset(train)
    logger.debug(train.head())
    ml_model_dir = MODEL_DIR / "autogluon"
    time_limits = MAX_TRAIN_SECS
    metric = "roc_auc"
    predictor = task.fit(
        train_data=train,
        label=Y_TARGET,
        output_directory=ml_model_dir,
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
    mlflow.log_param("max_secs", MAX_TRAIN_SECS)
    mlflow.log_metric("val_auc", val_auc)
    mlflow.log_metric("test_auc", test_auc)

    optimize_for_deploy(predictor)

    make_archive(ml_model_dir, "zip", root_dir=ml_model_dir)
    ml_model_zip = str(ml_model_dir) + ".zip"
    return ml_model_zip
