import logging

from h2o.frame import H2OFrame

logging.basicConfig(level=logging.INFO)
import tempfile

import mlflow
import mlflow.h2o
import mlflow.pyfunc
from h2o.automl import H2OAutoML

logger = logging.getLogger(__name__)
logger.debug("h2o_mlflow")


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
