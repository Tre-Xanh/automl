# %%
from automl.deploy import build_image, deploy_image, get_workspace
import os

from dotenv import load_dotenv

from automl.config import PRJ_DIR

load_dotenv(dotenv_path=PRJ_DIR / ".trained.env", override=True)
model_uri = os.getenv("MLFLOW_MODEL")
model_uri
# %%

exp_name = "congvc2-mlflow-azureml"

ws = get_workspace()
model_image = build_image(model_uri, exp_name, ws)
# %%
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
from loguru import logger


def deploy_image(model_image, exp_name, ws: Workspace):
    logger.info("model_image {}", model_image)
    websvc_deploy_cfg = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=5)
    websvc = Webservice.deploy_from_image(
        image=model_image,
        name=exp_name,
        deployment_config=websvc_deploy_cfg,
        workspace=ws,
        overwrite=True,
    )
    websvc.wait_for_deployment()
    logger.info(websvc.scoring_uri)
    return websvc


websvc = deploy_image(model_image, exp_name, ws)

# %%
