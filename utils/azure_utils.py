from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

subscription_id = ''
resource_group = ''
workspace_name = ''

# find azureml-examples/sdk/python documentation
# https://github.com/Azure/azureml-examples/tree/main/sdk/python

ml_client = MLClient(InteractiveBrowserCredential(), subscription_id, resource_group)

ws = ml_client.workspaces.get(workspace_name)
# uncomment this line after providing a workspace name above
print(ws.location,":", ws.resource_group)