from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, Data
from azure.ai.ml.constants import AssetTypes

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
        credential=credential, subscription_id=subscription_id, resource_group_name=resource_group, workspace_name=workspace_name,
    )

    return ml_client


ml_client = get_workspace()

# setup up data
train_path = 'azureml://subscriptions/5577d63d-1715-4e34-a55c-e70ee101434b/resourcegroups/Azure-ML/workspaces/azureml-luis/datastores/fruitclassification/paths/train/'

train_data = Data(
    path = train_path,
    type=AssetTypes.URI_FOLDER,
    description = "train URI folder from fruit classification data lake",
    name = "train_folder",
    version = '1'
)

ml_client.data.create_or_update(train_data)

""" fruit_env = Environment(
    name='fruit_env',
    conda_file='environment.yml'
) """