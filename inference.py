import torch
from models.model import PretrainedResNet
import torch
from PIL import Image
from utils.transforms import get_transforms
import numpy as np
from pathlib import Path

def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path)
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return transformed['image'].unsqueeze(0)  # Add batch dimension

def infer_image(image_path, model, device):
    # 1. Preprocessing
    _, test_transform = get_transforms()
    image_tensor = load_and_preprocess_image(image_path, test_transform)
    image_tensor = image_tensor.to(device)
    
    # 2. Model Forward Pass
    with torch.no_grad():  # Disables gradient computation
        model.eval()
        logits = model(image_tensor)

    # 3. Post-processing
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, probabilities[0, predicted_class].item()


# Example usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PretrainedResNet(num_classes=2)
model.to(device)
model.load_state_dict(torch.load("fruit_resnet.pth"))

image_path = "test_image.jpg"
predicted_class, confidence = infer_image(image_path, model, device)
print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")