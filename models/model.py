# file for pre-trained resnet architectures
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class PretrainedResNet(nn.Module):
    """
    Simple pretrained Resnet34 model with default weights. 
    Head has been changed for fine-tuning.
    """
    def __init__(self, num_classes):
        super(PretrainedResNet, self).__init__()
        self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)