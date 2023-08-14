from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment, Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes

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

path = 'azureml://datastores/fruitclassification/paths/fruit_classification_datasets'
data_type = AssetTypes.URI_FOLDER
mode = InputOutputModes.RO_MOUNT

inputs = {
    "input_data": Input(type=data_type, path=path, mode=mode)
}

command_job = command(
    code="./",
    command="python train.py --data_dir ${{inputs.input_data}}",
    inputs=inputs,
    environment="fruit_env@latest",
    compute="cpu-cluster",
    name=f"fruit_classification_training"
)

ml_client.jobs.create_or_update(command_job)