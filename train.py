"""
Training script for RoshamboNet on 64x64 Pseudo-Event Motion Mask Dataset.
Optimized for ultra-fast CPU training (10-15 epochs in <15 seconds).
Saves trained model weights to motion_model.pth.
"""

import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets, transforms

from model import RoshamboNet, CLASS_NAMES


def get_data_loaders(dataset_dir: str, batch_size: int = 32, val_split: float = 0.2):
    """
    Loads 64x64 images from dataset directory with train/validation split and data augmentations.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist. Run collect_data.py first!")

    # Transforms for 64x64 single-channel motion masks
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),  # Scales [0, 255] to [0.0, 1.0]
    ])

    val_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    base_dataset = datasets.ImageFolder(root=dataset_dir)
    total_samples = len(base_dataset)
    if total_samples == 0:
        raise ValueError(f"No images found in '{dataset_dir}'. Collect data before training.")

    print(f"[train] Found {total_samples} total samples across classes: {base_dataset.classes}")

    # Stratified or randomized train-val split
    indices = list(range(total_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    split_idx = int(np.floor(val_split * total_samples))
    val_indices, train_indices = indices[:split_idx], indices[split_idx:]

    train_dataset = datasets.ImageFolder(root=dataset_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=dataset_dir, transform=val_transform)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=2, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=2, pin_memory=False)

    print(f"[train] Training samples: {len(train_indices)} | Validation samples: {len(val_indices)}")
    return train_loader, val_loader, len(train_indices), len(val_indices)


def train_model(args):
    """Main training routine."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu")
    print(f"[train] Execution device: {device} (Optimized for ultra-low latency CPU inference)")

    # Data loaders
    train_loader, val_loader, num_train, num_val = get_data_loaders(
        args.dataset_dir, batch_size=args.batch_size, val_split=args.val_split
    )

    # Initialize RoshamboNet
    model = RoshamboNet(num_classes=len(CLASS_NAMES), pooling="avg").to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] RoshamboNet initialized. Total trainable parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": []
    }

    start_train_time = time.time()
    best_val_acc = 0.0

    print(f"\n================ Starting Training ({args.epochs} Epochs) ================")
    for epoch in range(1, args.epochs + 1):
        # ---------------- Training Phase ----------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        scheduler.step()
        epoch_train_loss = running_loss / max(total_train, 1)
        epoch_train_acc = correct_train / max(total_train, 1)

        # ---------------- Validation Phase ----------------
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_val += torch.sum(preds == labels.data).item()
                total_val += labels.size(0)

        epoch_val_loss = running_val_loss / max(total_val, 1)
        epoch_val_acc = correct_val / max(total_val, 1)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc*100:5.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:5.2f}%")

        if epoch_val_acc >= best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), args.output_model)

    train_duration = time.time() - start_train_time
    print(f"\n[train] Training completed in {train_duration:.2f}s!")
    print(f"[train] Best Validation Accuracy: {best_val_acc*100:.2f}%")
    print(f"[train] Optimal model weights saved to: {os.path.abspath(args.output_model)}")

    # ---------------- Evaluation & Confusion Matrix ----------------
    evaluate_and_plot(model, val_loader, device, history, args.metrics_plot)


def evaluate_and_plot(model, val_loader, device, history, plot_path):
    """Calculates confusion matrix and saves training performance curve plot."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.numpy())

    if len(all_targets) > 0:
        print("\n--- Detailed Classification Report ---")
        present_classes = sorted(list(set(all_targets) | set(all_preds)))
        target_names = [CLASS_NAMES[i] for i in present_classes]
        print(classification_report(all_targets, all_preds, labels=present_classes, target_names=target_names, zero_division=0))
        
        cm = confusion_matrix(all_targets, all_preds, labels=present_classes)
        print("Confusion Matrix:")
        print(cm)

    # Plot metrics
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss", color="crimson")
    plt.plot(history["val_loss"], label="Val Loss", color="royalblue", linestyle="--")
    plt.title("Loss vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc", color="crimson")
    plt.plot(history["val_acc"], label="Val Acc", color="royalblue", linestyle="--")
    plt.title("Accuracy vs. Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"[train] Training curve plot saved to: {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RoshamboNet on 64x64 Pseudo-Event Masks")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Dataset directory path")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs (10-15 recommended)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation dataset fraction")
    parser.add_argument("--output_model", type=str, default="motion_model.pth", help="Target model weights filename")
    parser.add_argument("--metrics_plot", type=str, default="training_metrics.png", help="Path to save metric plot")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU training")
    args = parser.parse_args()

    train_model(args)
