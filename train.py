import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.model import PretrainedResNet
from utils.transforms import get_transforms
from utils.dataset import FruitDataset
from utils.utils import get_image_paths
import copy


def train_model(
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    save_checkpoint=False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_accuracy = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_predictions / total_predictions

        print(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.4f}"
        )

        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_predictions = 0

        with torch.no_grad():
            for val_images, val_labels in valid_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_images)
                val_loss = criterion(val_outputs, val_labels)

                val_running_loss += val_loss.item() * val_images.size(0)
                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total_predictions += val_labels.size(0)
                val_correct_predictions += (val_predicted == val_labels).sum().item()

        val_epoch_loss = val_running_loss / len(valid_loader.dataset)
        val_epoch_accuracy = val_correct_predictions / val_total_predictions

        print(
            f"Validation Loss: {val_epoch_loss:.4f} - Validation Accuracy: {val_epoch_accuracy:.4f}"
        )

        if val_epoch_accuracy > best_accuracy:
            best_accuracy = val_epoch_accuracy
            best_model_weights = copy.deepcopy(model.state_dict())

        scheduler.step()

    print("Training complete!")
    print(f"Best Validation Accuracy: {best_accuracy:.4f}")

    model.load_state_dict(best_model_weights)

    if save_checkpoint:
        torch.save(model.state_dict(), "fruit_resnet.pth")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_checkpoint", type=bool, default=True)

    args = parser.parse_args()

    # Load architecture
    model = PretrainedResNet(num_classes=2)

    # Get dataset paths
    train_image_paths, valid_image_paths, test_image_paths, classes = get_image_paths(
        data_dir=args.data_dir
    )
    # print(len(train_image_paths), len(valid_image_paths), len(test_image_paths), len(classes))

    # Get transforms
    train_transforms, test_transforms = get_transforms()

    # Create datasets
    train_dataset = FruitDataset(train_image_paths, train_transforms)
    valid_dataset = FruitDataset(valid_image_paths, test_transforms)
    test_dataset = FruitDataset(test_image_paths, test_transforms)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    # Training parameters
    device = torch.device("mps" if torch.has_mps else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    num_epochs = args.n_epochs

    # Training Loop
    trained_model = train_model(
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        save_checkpoint=args.save_checkpoint,
    )
