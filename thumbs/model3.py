from __future__ import annotations

from thumbs.assets import BASE_DIR, MODEL_PATH, ConvNeXtClassifier, INFERENCE_TRANSFORM, get_device, load_class_names, save_class_names
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from typing import Sequence, Tuple
from torchvision import datasets
from collections import Counter
import torch.optim as optim
import torch.nn as nn
import PIL.Image
import torch
import sys


DATASET_DIR = BASE_DIR / "train"
DEVICE = get_device()


TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.GaussianBlur(3),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


EVAL_TRANSFORM = INFERENCE_TRANSFORM


class TransformedSubset(Dataset):
    """Wrap a subset to apply transforms without mutating the base dataset."""

    def __init__(self, subset: Dataset, transform: transforms.Compose) -> None:
        # Store the subset reference and the transform pipeline for later use.
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        # Mirror the subset length so DataLoader knows how many samples exist.
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.subset[idx]
        # Return the transformed item so augmentation happens on-the-fly.
        # Apply the composed transforms only when one was supplied.
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def prepare_dataloaders(
    batch_size: int = 16,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, Sequence[str]]:
    # Create augmented train/validation loaders and return associated classes.
    # Bail out early if the training directory is missing.
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")

    base_dataset = datasets.ImageFolder(root=str(DATASET_DIR))
    # Require at least two images so the split can produce non-empty subsets.
    if len(base_dataset) < 2:
        raise RuntimeError("Dataset must contain at least two images for train/val splitting.")

    class_names = base_dataset.classes
    print(f"Classes found: {class_names}")
    print(f"Number of images: {len(base_dataset)}")
    label_counts = Counter(base_dataset.targets)
    readable_counts = {class_names[idx]: count for idx, count in label_counts.items()}
    print(f"Label distribution: {readable_counts}")

    train_size = int((1.0 - val_split) * len(base_dataset))
    val_size = len(base_dataset) - train_size
    # Validate that both train and validation splits will have samples.
    if train_size <= 0 or val_size <= 0:
        raise RuntimeError("Invalid train/validation split. Add more images to the dataset.")

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(
        base_dataset, [train_size, val_size], generator=generator
    )

    train_dataset = TransformedSubset(train_subset, TRAIN_TRANSFORM)
    val_dataset = TransformedSubset(val_subset, EVAL_TRANSFORM)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, class_names


# Construct the ConvNeXt classifier and optionally warm start from disk.
def build_model(num_classes: int) -> ConvNeXtClassifier:
    model = ConvNeXtClassifier(num_classes=num_classes).to(DEVICE)
    # Reload saved weights when available so training can resume.
    if MODEL_PATH.exists():
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            print(f"Loaded existing weights from '{MODEL_PATH.name}'.")
        except Exception as exc:
            print(f"Warning: unable to load existing weights ({exc}). Starting fresh.")
    return model


# Full training loop with early stopping and checkpointing.
def train_model(epochs: int = 15, patience: int = 3) -> None:
    try:
        train_loader, val_loader, class_names = prepare_dataloaders()
    except Exception as exc:
        print(f"Failed to prepare data loaders: {exc}")
        return

    save_class_names(class_names)

    model = build_model(num_classes=len(class_names))
    # Optimize only the layers that remain trainable.
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(trainable_params, lr=8e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_loss = float("inf")
    early_stop_counter = 0

    # Iterate over epochs while tracking best validation loss.
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs} started...")

        model.train()
        running_loss = 0.0
        # Standard training loop across all mini-batches.
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).long()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            # Evaluate on the validation loader without gradient tracking.
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Avoid division by zero if the validation loader is empty.
        val_accuracy = 100 * correct / total if total > 0 else 0.0
        avg_train_loss = running_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)

        print(
            f"Epoch {epoch + 1} complete. "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )

        # Update the best checkpoint when validation loss improves.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new best model to '{MODEL_PATH.name}'.")
        else:
            early_stop_counter += 1
            # Stop training if loss has not improved within `patience`.
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

        scheduler.step()


def load_model_for_inference() -> Tuple[ConvNeXtClassifier, Sequence[str]]:
    # Load trained weights and class labels strictly for inference.
    class_names = load_class_names()
    # Fail loudly if trained weights are missing.
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}. Train the model before prediction."
        )

    model = ConvNeXtClassifier(num_classes=len(class_names)).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names


def predict_image(image_path: str) -> None:
    # Load a still image and print class probabilities for inspection.
    try:
        model, class_names = load_model_for_inference()
    except Exception as exc:
        print(f"Unable to load model assets: {exc}")
        return

    try:
        image = PIL.Image.open(image_path).convert("RGB")
    except (FileNotFoundError, OSError) as exc:
        print(f"Failed to open image '{image_path}': {exc}")
        return

    tensor = EVAL_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_prob, top_class = torch.max(probabilities, dim=0)

    predicted_label = class_names[top_class.item()]
    print(f"Predicted class: {predicted_label} (confidence: {top_prob.item() * 100:.2f}%)")
    print("Class probabilities:")
    # Print probability for each class in display order.
    for idx, prob in enumerate(probabilities):
        print(f"{class_names[idx]}: {prob.item() * 100:.2f}%")


# Allow the module to act as both a trainer and a predictor from CLI.
if __name__ == "__main__":
    # Use prediction mode when a path was provided on the command line.
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        train_model()
