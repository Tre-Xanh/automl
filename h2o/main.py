# %%
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import tempfile

import h2o
import mlflow
import mlflow.pyfunc
import pandas as pd
from h2o import H2OFrame
from sklearn.model_selection import train_test_split

from learner import Learner
from preproc import Preproc


class Predictor(mlflow.pyfunc.PythonModel):
    def load_context(self, context):

        h2o.init()
        logger.info(f"artifacts {context.artifacts}")
        self.h2o_model = h2o.load_model(context.artifacts["h2o_model_path"])
        self.prep_model = Preproc.load_model(context.artifacts["prep_model_path"])

    def predict(self, context, model_input):
        # Convert input from Pandas
        input = self.prep_model.transform(model_input)
        output: H2OFrame = self.h2o_model.predict(input)

        # Convert output to Pandas
        return output.as_data_frame()


# %%
def main():
    mlflow.start_run()

    # %% 実験データのダウンロード
    data_uri: str = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
    data = pd.read_csv(data_uri)
    logger.debug("Downloaded data")

    # %% 学習・テストデータの分割
    dftrain: pd.DataFrame
    dftest: pd.DataFrame
    dftrain, dftest = train_test_split(data)
    logger.info(f"train_test_split {data.shape} -> {dftrain.shape} + {dftest.shape}")

    # %% データ前処理
    prep = Preproc()
    train: H2OFrame = prep.fit_transform(dftrain)
    prep_model_path = prep.save_model()
    test: H2OFrame = prep.transform(dftest)

    # %% 機械学習 AutoML
    h2o_model_path = Learner().fit(train, test)

    # %% MLflowで学習済みの前処理・モデルを保存
    conda_env = "./conda.yaml"
    code_path = ["."]  # 重要：ローカル Python ソースコードの所在地
    artifacts = dict(
        prep_model_path=prep_model_path,
        #
        h2o_model_path=h2o_model_path,
    )
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=Predictor(),
        code_path=code_path,
        artifacts=artifacts,
        conda_env=conda_env,
    )

    main_model_path = os.path.join(tempfile.mkdtemp(), "main.model")
    mlflow.pyfunc.save_model(
        path=main_model_path,
        python_model=Predictor(),
        code_path=code_path,
        artifacts=artifacts,
        conda_env=conda_env,
    )

    # %% MLflowとH2OAutoMLで保存したモデルをリロードして予測結果を比較
    reloaded_mlflow = mlflow.pyfunc.load_model(main_model_path)
    predictions_mlflow: pd.DataFrame = reloaded_mlflow.predict(dftest)
    logger.info(f"predictions_mlflow\n{predictions_mlflow}")

    reloaded_h2o = h2o.load_model(h2o_model_path)
    predictions_h2o = reloaded_h2o.predict(test).as_data_frame()
    logger.info(f"predictions_h2o\n{predictions_h2o}")

    assert predictions_h2o.equals(predictions_mlflow)

    # Save test dataframe for later API test
    dftest.to_csv("dftest.csv", index=False)
    predictions_mlflow.to_csv("predictions_mlflow.csv", index=False)

    # Log final model path
    logger.debug(f"prep_model_path {prep_model_path}")
    logger.debug(f"h2_model_path {h2o_model_path}")
    logger.info(
        f"""### Run the saved model as ###
    MODEL={main_model_path}
    mlflow models serve -m $MODEL
    """
    )

    mlflow.end_run()


# %%
if __name__ == "__main__":
    main()
