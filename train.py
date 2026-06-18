import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm
from pathlib import Path
import argparse

from src.models import EnsembleDetector, EfficientNetDetector
from src.preprocessing import DeepfakeDataset, get_training_transforms, get_validation_transforms
from src.utils.metrics import compute_metrics, plot_roc_curve, plot_confusion_matrix


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing,
        )
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_score):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


def train_one_epoch(model, dataloader, criterion, optimizer, device, use_amp=False):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].detach().cpu().numpy())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    for images, labels in tqdm(dataloader, desc="Validating", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        loss = criterion(logits, labels)

        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics, np.array(all_labels), np.array(all_probs), np.array(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per class for quick trial (e.g. 500)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    args = parser.parse_args()

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    if args.epochs:
        config["training"]["epochs"] = args.epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Using device: {device}")
    if not use_amp:
        print("Running on CPU — mixed precision disabled")

    train_transform = get_training_transforms(config["data"]["image_size"])
    val_transform = get_validation_transforms(config["data"]["image_size"])

    train_dataset = DeepfakeDataset("data/processed", split="train", transform=train_transform)
    val_dataset = DeepfakeDataset("data/processed", split="val", transform=val_transform)

    if args.max_samples:
        n = min(args.max_samples * 2, len(train_dataset))
        train_dataset = Subset(train_dataset, list(range(n)))
        n_val = min(args.max_samples, len(val_dataset))
        val_dataset = Subset(val_dataset, list(range(n_val)))
        print(f"Trial mode: {len(train_dataset)} train, {len(val_dataset)} val samples")

    print(f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val")

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
        pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    if config["model"]["ensemble"]["enabled"]:
        model = EnsembleDetector(config).to(device)
        print("Model: Ensemble (4 models)")
    else:
        model = EfficientNetDetector(
            model_name=config["model"]["backbone"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        ).to(device)
        print(f"Model: {config['model']['backbone']}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    criterion = FocalLoss(
        alpha=0.25, gamma=2.0,
        label_smoothing=config["training"]["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    os.makedirs("weights", exist_ok=True)
    best_auc = 0.0
    epochs = config["training"]["epochs"]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 60)

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp)

        val_metrics, y_true, y_prob, y_pred = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Train — Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']:.4f} | AUC: {train_metrics.get('auc_roc', 0):.4f}")
        print(f"Val   — Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | AUC: {val_metrics.get('auc_roc', 0):.4f} | EER: {val_metrics.get('eer', 0):.4f}")

        val_auc = val_metrics.get("auc_roc", val_metrics["accuracy"])
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_auc": best_auc,
                "config": config,
            }, "weights/best_model.pth")
            print(f"  → Saved best model (AUC: {best_auc:.4f})")

        early_stopping(val_auc)
        if early_stopping.should_stop:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print("\nGenerating evaluation plots...")
    plot_roc_curve(y_true, y_prob)
    plot_confusion_matrix(y_true, y_pred)
    print("Training complete!")


if __name__ == "__main__":
    main()
