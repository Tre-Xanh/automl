import azureml.core
import mlflow
import mlflow.azureml
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice
from loguru import logger


def get_workspace():
    ws = Workspace.from_config()
    logger.debug("AzureML SDK version: {}", azureml.core.VERSION)
    logger.debug("MLflow version: {}", mlflow.version.VERSION)
    logger.info(
        f"""
{ws.name}
{ws.resource_group}
{ws.location}
{ws.subscription_id}
"""
    )
    return ws


def build_image(model_uri, exp_name: str, ws: Workspace):
    logger.info("model {}", model_uri)
    model_image, _ = mlflow.azureml.build_image(
        model_uri=model_uri,
        workspace=ws,
        model_name=exp_name,
        image_name=exp_name,
        synchronous=True,
    )
    return model_image


def deploy_image(model_image, exp_name, ws: Workspace):
    logger.info("image {}", model_image)
    websvc_deploy_cfg = AciWebservice.deploy_configuration()
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


# https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments
# from azureml.core.environment import Environment

# azure_env = Environment.from_conda_specification(
#     name=expName, file_path="conda.yml"
# )
# azure_env.docker.enabled = True
# azure_env
# azure_env.register(workspace=ws)
# from azureml.core import Image
# build = azure_env.build(workspace=ws)  # build on Azure
# # build = azure_env.build_local(workspace=ws, useDocker=True, pushImageToWorkspaceAcr=True)
# build.wait_for_completion(show_output=True)
