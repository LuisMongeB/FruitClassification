from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

subscription_id = ''
resource_group = ''
workspace_name = ''

ml_client = MLClient(InteractiveBrowserCredential(), subscription_id, resource_group)

ws = ml_client.workspaces.get(workspace_name)
# uncomment this line after providing a workspace name above
print(ws.location,":", ws.resource_group)