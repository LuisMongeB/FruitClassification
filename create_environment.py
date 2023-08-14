from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import DefaultAzureCredential

from dotenv import load_dotenv
import os

load_dotenv()

def get_workspace(verbose=False):

    load_dotenv()

    subscription_id = os.environ['SUBSCRIPTION_ID']
    resource_group = os.environ['RESOURCE_GROUP']
    workspace_name = os.environ['WORKSPACE_NAME']
    credential = DefaultAzureCredential()

    if verbose:
        print(f"Resource Group:  {resource_group} | Subscription: {subscription_id} | {workspace_name}")

    ml_client = MLClient(
        credential=credential, subscription_id=subscription_id, resource_group_name=resource_group, workspace_name=workspace_name,
    )

    return ml_client

ml_client = get_workspace()

print("Beginning creation of Azure ML environment...")
env_docker_context = Environment(
    build=BuildContext(path="docker-context"),
    name="fruit_env",
    description="Environment created from a Docker context.",
)
ml_client.environments.create_or_update(env_docker_context)
print("Done creating environment!")