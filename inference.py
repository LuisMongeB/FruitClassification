import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from utils.dataset import Fruits

from pathlib import Path
import pandas as pd

def prepare_resnet18(checkpoint):
        # Load pre-trained model
        model = resnet18(weights=None)
        model.to(device)

        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.fc = model.fc.to(device)

        model.load_state_dict(torch.load(checkpoint))

        return model


def create_inference_dataset(path_to_images):
    
    fruit_dir = Path(path_to_images)

    # select all files with 'uchua' in the name and put them in a list
    uchua = [x.name for x in fruit_dir.iterdir() if x.is_file() and 'uchua' in x.name]
    mora = [x.name for x in fruit_dir.iterdir() if x.is_file() and 'mora' in x.name]

    # create a pandas dataframe with the file names and the labels encoded as 0 and 1
    fruits_inference = pd.DataFrame({'file': uchua + mora, 'label': [0]*len(uchua) + [1]*len(mora)})

    fruits_inference.to_csv('fruits_inference.csv', index=False)
    return

def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        input = input.to(device)
        target = target.to(device)
        pred = model(input)
        m = nn.Softmax(dim=1)
        pred = m(pred)
        predicted_index = pred[0].argmax(0)
        predicted = class_mapping[predicted_index.item()]
        expected = class_mapping[target.item()]
        return predicted, expected


if __name__ == '__main__':
    
    # class mapping
    class_mapping = {0: 'uchua', 1: 'mora'}
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this will be 'cpu'
    
    # load model
    checkpoint = 'resnet18.pth.tar'
    model = prepare_resnet18(checkpoint=checkpoint)
    print(model)

    # create inference dataset
    if not Path('fruits_inference.csv').exists():
        create_inference_dataset(path_to_images='./fruits_inference')
    
    # load inference dataset
    dataset = Fruits(csv_file='fruits_inference.csv', root_dir='./fruits_inference',
                    transform=transforms.ToTensor())
    inference_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)
    # get a sample
    correct = 0
    for i in range(len(dataset)):
         
        input, target = next(iter(inference_loader))[0], next(iter(inference_loader))[1]

        # predict
        predicted, expected = predict(model, input, target, 
                                    class_mapping)
        
        if predicted == expected:
            correct += 1
        # print(f'Predicted: {predicted}, Expected: {expected}')
    print(f'Correct: {correct} | Accuracy: {correct/len(dataset)}')




