"""Train the PlantCNN baseline.

Trains (or validates) a small CNN on PlantVillage tomato classes with early
stopping and saves the best checkpoint.
"""

import sys
import os
sys.path.append(os.getcwd())

from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from utils import (
    PlantCNN,
    DROPOUT,
    MODEL_DIR,
    NUM_CLASSES,
    TRAIN_DIR,
    VAL_DIR,
    get_device,
    get_transform,
)

# TRAINING SETTINGS
BATCH_SIZE: int = 256
EPOCHS: int = 20
LR: float = 1e-3
PATIENCE: int = 5
MIN_DELTA: float = 0.1


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    criterion: nn.Module,
) -> dict[str, Any]:
    """Compute loss, accuracy, and per-class metrics on a data loader.

    Args:
        model: Model to evaluate.
        loader: Data loader for evaluation.
        device: Torch device for inference.
        num_classes: Number of classes in the classifier head.
        criterion: Loss function to compute average loss.
    Returns:
        Dictionary with loss, accuracy, per-class metrics, and confusion matrix.
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            for t, p in zip(labels.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

    avg_loss = total_loss / max(1, total)
    acc = 100 * correct / max(1, total)

    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp

    precision = torch.where(tp + fp > 0, tp / (tp + fp), torch.zeros_like(tp))
    recall = torch.where(tp + fn > 0, tp / (tp + fn), torch.zeros_like(tp))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(tp),
    )

    macro_f1 = f1.mean().item()

    return {
        "loss": avg_loss,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "cm": cm,
    }


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    if not TRAIN_DIR.exists() or not VAL_DIR.exists():
        raise FileNotFoundError(
            f"Missing dataset folders. Expected {TRAIN_DIR} and {VAL_DIR}. "
            "Run image_extension/00_data_preparation.py first."
        )

    transform = get_transform()

    train_data = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    val_data = datasets.ImageFolder(VAL_DIR, transform=transform)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader


def train() -> None:
    """Train the CNN and save the best checkpoint by validation accuracy."""
    print("========== Train CNN ==========")

    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders()

    model = PlantCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0

        train_bar = tqdm(train_loader, desc=f"Train {epoch}/{EPOCHS}", leave=False)
        for batch_idx, (images, labels) in enumerate(train_bar, start=1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            avg_loss = train_loss / batch_idx
            train_bar.set_postfix(loss=f"{avg_loss:.3f}")

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Val {epoch}/{EPOCHS}", leave=False)
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                batch_size = labels.size(0)
                total += batch_size
                correct += (predicted == labels).sum().item()
                val_loss += loss.item() * batch_size
                acc = 100 * correct / max(1, total)
                val_bar.set_postfix(acc=f"{acc:.2f}%")

        avg_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, total)
        val_acc = 100 * correct / max(1, total)
        print(
            f"Epoch {epoch}/{EPOCHS} - Train Loss: {avg_loss:.3f} - "
            f"Val Loss: {avg_val_loss:.3f} - Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_DIR / "plant_cnn.pt")
            print(f"Saved best model (acc={val_acc:.2f}%)")
        elif val_acc < best_acc + MIN_DELTA:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{PATIENCE} epochs")
            if epochs_no_improve >= PATIENCE:
                print("Early stopping triggered")
                break

    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")

    if (MODEL_DIR / "plant_cnn.pt").exists():
        best_model = PlantCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
        best_model.load_state_dict(torch.load(MODEL_DIR / "plant_cnn.pt", map_location=device))
    else:
        best_model = model

    final = evaluate_metrics(best_model, val_loader, device, NUM_CLASSES, criterion)
    class_names = val_loader.dataset.classes
    f1_items = ", ".join(
        f"{name}={score:.2f}" for name, score in zip(class_names, final["f1"].tolist())
    )
    print(
        f"Best model val: loss={final['loss']:.3f}, acc={final['acc']:.2f}%, "
        f"macro_f1={final['macro_f1']:.3f}"
    )
    print(f"Per-class F1: {f1_items}")
    print(f"Confusion matrix: {final['cm'].tolist()}")


def main() -> None:
    """Entry point."""
    model_path = MODEL_DIR / "plant_cnn.pt"
    if model_path.exists():
        print(f"Found existing model at {model_path}. Running validation only.")
        device = get_device()
        print(f"Using device: {device}")
        _, val_loader = build_dataloaders()
        model = PlantCNN(num_classes=NUM_CLASSES, dropout=DROPOUT).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        criterion = nn.CrossEntropyLoss()
        final = evaluate_metrics(model, val_loader, device, NUM_CLASSES, criterion)
        class_names = val_loader.dataset.classes
        f1_items = ", ".join(
            f"{name}={score:.2f}" for name, score in zip(class_names, final["f1"].tolist())
        )
        print(
            f"Val: loss={final['loss']:.3f}, acc={final['acc']:.2f}%, "
            f"macro_f1={final['macro_f1']:.3f}"
        )
        print(f"Per-class F1: {f1_items}")
        print(f"Confusion matrix: {final['cm'].tolist()}")
    else:
        train()


if __name__ == "__main__":
    main()
