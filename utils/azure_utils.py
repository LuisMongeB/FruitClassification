from dotenv import load_dotenv
import os

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute

load_dotenv()

# find azureml-examples/sdk/python documentation
# https://github.com/Azure/azureml-examples/tree/main/sdk/python

subscription_id = os.environ['SUBSCRIPTION_ID']
resource_group = os.environ['RESOURCE_GROUP']
workspace_name = os.environ['WORKSPACE_NAME']

# Connect to workspace
ml_client = MLClient(DefaultAzureCredential(),
                    subscription_id,
                    resource_group,
                    workspace_name)

# uncomment this line after providing a workspace name above
print("Welcome to your workspace: ", ml_client.workspace_name)
for datastore in ml_client.datastores.list():
    print(f"These are the datastores: {datastore}\n")

# datastore to choose: 'container_name': 'azureml-blobstore-ee259bd0-6cb1-43d5-996e-855852ad4506', 'account_name': 'azuremlluis6635064614'

# create compute if not exists
cpu_compute_target = 'cpu-cluster'

try:
    ml_client.compute.get(cpu_compute_target)
    print("Compute: ", cpu_compute_target)
except Exception:
    print("Creating new cpu compute target...")
    compute = AmlCompute(
        name=cpu_compute_target,
        size="STANDARD_D2_V2",
        min_instances=0,
        max_instances=2
    )
    ml_client.compute.begin_create_or_update(compute).result()
    print("Done!")