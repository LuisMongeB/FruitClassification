from pillow_heif import register_heif_opener
from PIL import Image
import os
import cv2
from pandas.core.common import flatten
import numpy as np
import matplotlib.pyplot as plt
import glob
import random
from copy import deepcopy
import albumentations as A
from albumentations.pytorch import ToTensorV2

def heic2jpg(heic_dir, jpg_dir):
    """
    Converts HEIC images to JPG format.
    
    Args:
        heic_dir (str): Directory path containing HEIC images.
        jpg_dir (str): Directory path to save the converted JPG images.
    """
    register_heif_opener()
    heic_files = [image for image in os.listdir(heic_dir) if '.heic' in image]
    
    for image in heic_files:
        tmp_img = Image.open(os.path.join(heic_dir, image))
        jpg_image = image.replace('.heic', '.jpg')
        if not os.path.exists(jpg_dir):
            os.makedirs(jpg_dir, exist_ok=True)
        tmp_img.save(os.path.join(jpg_dir, jpg_image))


def resize_images(image_dir, dim):
    """
    Resizes images to the specified dimensions.
    
    Args:
        image_dir (str): Directory path containing images to be resized.
        dim (int): Dimensions to resize the images (both width and height).
    """
    image_dir_paths = [image for image in os.listdir(image_dir) if image.endswith('.jpg')]
    for image_path in image_dir_paths:
        # opening image
        image = cv2.imread(os.path.join(image_dir, image_path))
        # resizing image
        resized_image = cv2.resize(image, [dim, dim])
        # saving image
        cv2.imwrite(os.path.join(image_dir, image_path), resized_image)

def load_image_paths(train_data_path, test_data_path):
    """
    Load image paths for training, validation, and testing datasets.
    
    Args:
        train_data_path (str): Path to the train data directory.
        test_data_path (str): Path to the test data directory.
    
    Returns:
        train_image_paths (list): List of image paths for the training dataset.
        valid_image_paths (list): List of image paths for the validation dataset.
        test_image_paths (list): List of image paths for the testing dataset.
    """
    train_image_paths = [] # to store image paths in a list
    classes = [] # to store class values

    # 1. Get all the paths from train_data_path and append image paths and class to respective lists
    for data_path in glob.glob(train_data_path + '/*'):
        classes.append(data_path.split('/')[-1]) 
        train_image_paths.append(glob.glob(data_path + '/*'))

    train_image_paths = list(flatten(train_image_paths))
    random.shuffle(train_image_paths)

    print('train_image_path example:', train_image_paths[0])
    print('class example:', classes[0])

    # 2. Split train valid from train paths (80,20)
    train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):]

    # 3. Create the test_image_paths
    test_image_paths = []
    for data_path in glob.glob(test_data_path + '/*'):
        test_image_paths.append(glob.glob(data_path + '/*'))

    test_image_paths = list(flatten(test_image_paths))

    print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))

    return train_image_paths, valid_image_paths, test_image_paths, classes


def visualize_augmentations(dataset, idx_to_class, image_paths, idx=0, samples=10, cols=5, random_img = False):
    """
    Visualizes augmented images from a dataset.
    
    Args:
        dataset (Dataset): Dataset object containing the images.
        idx_to_class (dict): Dictionary mapping class indices to class labels.
        image_paths (list): List of image paths in the dataset.
        idx (int): Index of the image to start visualizing from (default: 0).
        samples (int): Number of samples to visualize (default: 10).
        cols (int): Number of columns in the visualization grid (default: 5).
        random_img (bool): Whether to select images randomly from the dataset (default: False).
    """
    dataset = deepcopy(dataset)
    #we remove the normalize and tensor conversion from our augmentation pipeline
    dataset.transform = A.Compose([t for t in dataset.transform if not isinstance(t, (A.Normalize, ToTensorV2))])
    rows = samples // cols
    
        
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 8))
    for i in range(samples):
        if random_img:
            idx = np.random.randint(1,len(image_paths))
        image, lab = dataset[idx]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_axis_off()
        ax.ravel()[i].set_title(idx_to_class[lab])
    plt.tight_layout(pad=1)
    plt.show()