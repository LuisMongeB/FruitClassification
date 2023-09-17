from dotenv import load_dotenv
import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, BuildContext


def get_workspace(verbose=False):
    load_dotenv()

    subscription_id = os.environ["SUBSCRIPTION_ID"]
    resource_group = os.environ["RESOURCE_GROUP"]
    workspace_name = os.environ["WORKSPACE_NAME"]
    credential = DefaultAzureCredential()

    if verbose:
        print(
            f"Resource Group:  {resource_group} | Subscription: {subscription_id} | {workspace_name}"
        )

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    return ml_client


def create_enviroment(ml_client, docker_context_path, name):
    """
    MLClient class should be already instatiated.
    Creates an Environment in the AzureML workspace given a docker context path and a name.
    """
    print("Beginning creation of Azure ML environment...")
    env_docker_context = Environment(
        build=BuildContext(path=docker_context_path),
        name=name,
        description="Environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)
    print("Done creating environment!")


def create_compute_cluster():
    pass
