from torch.utils.data import Dataset
import cv2

class FruitDataset(Dataset):

    def __init__(self, image_paths, transforms=False):
        self.image_paths = image_paths
        self.idx_to_class = {i:j for i, j in enumerate(['uchua', 'mora'])}
        self.class_to_idx = {value:key for key, value in self.idx_to_class.items()}
        self.transform = transforms
        
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = image_filepath.split('/')[-2]
        label = self.class_to_idx[label]

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        return image, label
    
