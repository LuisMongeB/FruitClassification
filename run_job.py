import argparse
import random
import os
from azure.ai.ml import command, Input
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from utils.azure_utils import get_workspace

from dotenv import load_dotenv


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default=f'fruit_classification_training')
    parser.add_argument('--n_epochs', type=int, default=3)
    args = parser.parse_args()

    ml_client = get_workspace()

    dataset_path = os.environ['STORAGE_DATASET_PATH'] # path structure: azureml://datastores/<storage_account_name>/paths/<container_name>
    data_type = AssetTypes.URI_FOLDER # specifies that that data asset will be a directory, instead of a file
    mode = InputOutputModes.RO_MOUNT # only read necessary

    inputs = {
        "input_data": Input(type=data_type, path=dataset_path, mode=mode),
        "n_epochs": args.n_epochs
    }

    command_job = command(
        code="./",
        command="python train.py --data_dir ${{inputs.input_data}} --n_epochs ${{inputs.n_epochs}}", 
        inputs=inputs,
        environment="fruit_env@latest",
        compute="cpu-cluster",
        name=f"{args.experiment_name}_{args.n_epochs}epochs_{random.randint(4000, 50000)}"
    )

    ml_client.jobs.create_or_update(command_job)