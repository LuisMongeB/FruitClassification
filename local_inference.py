import torch
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
import os
import argparse

from utils.transforms import get_transforms
from utils.dataset import FruitDataset
from models.model import PretrainedResNet


def load_and_preprocess_image(image_path, transform):
    image = Image.open(image_path)
    image_np = np.array(image)
    transformed = transform(image=image_np)
    return transformed["image"].unsqueeze(0)  # Add batch dimension


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


def infer_directory(image_dir, model, device):
    image_paths = os.listdir(image_dir)
    predictions = {}
    for image in image_paths:
        prediction, confidence = infer_image(Path(image_dir, image), model, device)
        predictions[image] = [prediction, confidence]

    return predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PretrainedResNet(num_classes=2)
    model.to(device)
    model.load_state_dict(torch.load("fruit_resnet.pth"))

    image_dir = Path(args.input_dir)
    predictions = infer_directory(image_dir, model, device)
    pred_df = pd.DataFrame.from_dict(
        predictions, orient="index", columns=["Predictions", "Confidence"]
    ).reset_index()
    pred_df["correct_label"] = pred_df["index"].str.split("_", expand=True)[0]
    if not os.path.exists(f"{os.getcwd()}/outputs"):
        os.mkdir(f"{os.getcwd()}/outputs")
    pred_df.to_csv(Path(f"{os.getcwd()}/outputs/predictions_test.csv"))
    print(pred_df)
    return


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage="USAGE python inference.py --input_dir")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../datasets-backup/fruitclassification/test_images",
    )
    args = parser.parse_args()

    main()
