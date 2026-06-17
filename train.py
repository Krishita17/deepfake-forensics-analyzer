import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from pathlib import Path

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


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, gradient_clip=1.0):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()

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
    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = get_training_transforms(config["data"]["image_size"])
    val_transform = get_validation_transforms(config["data"]["image_size"])

    train_dataset = DeepfakeDataset("data/processed", split="train", transform=train_transform)
    val_dataset = DeepfakeDataset("data/processed", split="val", transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=config["data"]["num_workers"],
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"],
        shuffle=False, num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    if config["model"]["ensemble"]["enabled"]:
        model = EnsembleDetector(config).to(device)
    else:
        model = EfficientNetDetector(
            model_name=config["model"]["backbone"],
            num_classes=config["model"]["num_classes"],
            dropout=config["model"]["dropout"],
        ).to(device)

    criterion = FocalLoss(
        alpha=0.25, gamma=2.0,
        label_smoothing=config["training"]["label_smoothing"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    os.makedirs("weights", exist_ok=True)
    best_auc = 0.0

    for epoch in range(config["training"]["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 60)

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            config["training"]["gradient_clip"],
        )

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
    print("Training complete.")


if __name__ == "__main__":
    main()
