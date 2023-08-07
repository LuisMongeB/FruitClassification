from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

from dotenv import load_dotenv
from pathlib import Path
import os


def get_workspace(verbose=False):

    load_dotenv()

    subscription_id = os.environ['SUBSCRIPTION_ID']
    resource_group = os.environ['RESOURCE_GROUP']
    workspace_name = os.environ['WORKSPACE_NAME']
    credential = DefaultAzureCredential()

    if verbose:
        print(f"Resource Group:  {resource_group} | Subscription: {subscription_id} | {workspace_name}")

    ml_client = MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )

    return ml_client


ml_client = get_workspace()

fruit_env = Environment(
    name='fruit_env',
    conda_file='environment.yml'
)
