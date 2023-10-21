# Fruit Classifier with PyTorch!
Going back to visit my family after a long time of not being in Costa Rica. Fruits were maturing slowly over the weeks I was there. Decided to take pictures and use a pretrained ResNet CNN with PyTorch for doing image classification. 

This project offers a PyTorch-based solution to classify fruits utilizing Azure Machine Learning. The primary script configures and launches a training job on Azure ML's infrastructure. You can check a guide to certain aspects of this code in this [Medium](https://medium.com/@luisdmonge/azure-dp-100-prep-hands-on-with-pytorch-and-azure-ml-sdk-v2-8ab9497eb88f) article.

# Requirements
- Conda for environment creation
- Azure Subscription
- Azure Machine Learning Workspace

# Dataset
My dataset was composed of a couple hundred images of two fruits: black berries and golden berries. If you would like to have it for this project feel free to contact me.

However, I strongly recommend that you collect your own images of two distinct types fruits. During data collection you often learn very useful insights that will inform other phases of the ML life cycle.

The collected dataset should be stored in the following structure:

```
.
└── fruit_classification_datasets/
    ├── train/
    │   ├── fruit1/
    │   │   ├── fruit1_train_image1.jpg
    │   │   ├── fruit1_train_image2.jpg
    │   │   └── fruit2_train_image3.jpg
    │   └── fruit2/
    │       ├── fruit2_train_image1.jpg
    │       ├── fruit2_train_image2.jpg
    │       └── fruit2_train_image3.jpg
    ├── test/
    │   ├── fruit1/
    │   │   ├── fruit1_test_image1.jpg
    │   │   ├── fruit1_test_image2.jpg
    │   │   └── fruit1_test_image3.jpg
    │   └── fruit2/
    │       ├── fruit2_test_image1.jpg
    │       ├── fruit2_test_image2.jpg
    │       └── fruit2_test_image3.jpg
```

# Creating environment

In order to avoid dependency issues, you can create a a Conda environment using the following command:

```
conda create -n <ENV> python==3.10.12 
```
```
conda activate <ENV>  
```
```
pip install -r requirements.txt  
```
# Usage

To launch a job using Azure Machine Learning resources, you can run the following command in the terminal:
```
python run_job.py --experiment_name test_job --n_epochs 3 --batch_size 8 --save_checkpoint
```

If you want to launch the training locally, avoid run_job.py and go for train.py with the following command:

```
python train.py --data_dir PATH_TO_DATA_DIRECTORY --n_epochs N_EPOCHS --batch_size BATCH_SIZE
--save_checkpoint
```

# Environment Variables
Used python-dotenv in order to load environment variables without having to push them here. You can create a ```.env``` file in which to store:

-  SUBSCRIPTION_ID
-  RESOURCE_GROUP
-  WORKSPACE_NAME
-  STORAGE_DATASET_PATH

# Azure Configuration

Before executing the script, ensure that you've properly set up the Azure ML environment:

**Workspace:** Utilize the utils.azure_utils.get_workspace() function to fetch the Azure ML workspace.

**Data Asset:** The dataset is expected to be in the form of a directory (instead of a single file).

**Access Mode:** The script is configured to use RO_MOUNT which means it will only read the necessary data without making any alterations.

**Compute:** The training command is designed to run on the "cpu-cluster" compute target. Ensure you have this compute target set up in your workspace.

**Environment:** The script uses the "fruit_env@latest" Azure ML environment for training. This environment should be pre-configured with all necessary dependencies. For the creation of the enviornment, check out the blog post. 

# Contribution
Contributions, issues, and feature requests are welcome!
Give a ⭐️ if you like this project!

# Let's Connect
* On [Github](https://github.com/LuisMongeB) :star:
* On [Linkedin](https://www.linkedin.com/in/luis-diego-monge-bolanos/) :bulb: