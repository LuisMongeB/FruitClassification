# Fruit Classifier with PyTorch!
Going back to visit my family after a long time of not being in Costa Rica. Fruits were maturing slowly over the weeks I was there. Decided to take pictures and use a pretrained ResNet CNN with PyTorch for doing image classification. 

This project offers a PyTorch-based solution to classify fruits utilizing Azure Machine Learning. The primary script configures and launches a training job on Azure ML's infrastructure. You can check a guide to certain aspects of this code in [TEZT](URL)

# Requirements
- Conda for environment creation
- Azure Subscription
- Azure Machine Learning Workspace
- Azure Machine Learning Python SDK v2
- PyTorch

# Usage

To launch a job using Azure Machine Learning resources, you can run the following command in the terminal:
```
python run_job.py --experiment_name YOUR_EXPERIMENT_NAME --n_epochs YOUR_EPOCH_NUMBER --batch_size BATCH_SIZE --save_checkpoint
```

If you want to launch the training locally, avoid run_job.py and go for train.py with the following command:

```
python train.py --data_dir PATH_TO_DATA_DIRECTORY --n_epochs N_EPOCHS --batch_size BATCH_SIZE
--save_checkpoint
```

# Environment Variables
Used python-dotenv in order to load environment variables without having to push them here. You can create a ```.env``` file in which to store:

1. SUBSCRIPTION_ID
2. RESOURCE_GROUP
3. WORKSPACE_NAME
4. STORAGE_DATASET_PATH

# Azure Configuration

Before executing the script, ensure that you've properly set up the Azure ML environment:

**Workspace:** Utilize the utils.azure_utils.get_workspace() function to fetch the Azure ML workspace.

**Data Asset:** The dataset is expected to be in the form of a directory (instead of a single file).

**Access Mode:** The script is configured to use RO_MOUNT which means it will only read the necessary data without making any alterations.

**Compute:** The training command is designed to run on the "cpu-cluster" compute target. Ensure you have this compute target set up in your workspace.

**Environment:** The script uses the "fruit_env@latest" Azure ML environment for training. This environment should be pre-configured with all necessary dependencies.

