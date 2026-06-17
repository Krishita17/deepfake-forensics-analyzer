import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split="train", transform=None, balance=True):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        real_dir = self.root_dir / split / "real"
        fake_dir = self.root_dir / split / "fake"

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if real_dir.exists():
            for f in real_dir.iterdir():
                if f.suffix.lower() in extensions:
                    self.samples.append((str(f), 0))

        if fake_dir.exists():
            for f in fake_dir.iterdir():
                if f.suffix.lower() in extensions:
                    self.samples.append((str(f), 1))

        if balance and len(self.samples) > 0:
            self._balance_classes()

    def _balance_classes(self):
        real_samples = [s for s in self.samples if s[1] == 0]
        fake_samples = [s for s in self.samples if s[1] == 1]

        if not real_samples or not fake_samples:
            return

        min_count = min(len(real_samples), len(fake_samples))
        rng = np.random.RandomState(42)
        if len(real_samples) > min_count:
            real_samples = list(rng.choice(real_samples, min_count, replace=False))
        if len(fake_samples) > min_count:
            fake_samples = list(rng.choice(fake_samples, min_count, replace=False))

        self.samples = real_samples + fake_samples
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = cv2.imread(path)
        if image is None:
            raise RuntimeError(f"Failed to load image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class VideoDataset(Dataset):
    def __init__(self, root_dir, split="train", num_frames=16, transform=None):
        self.root_dir = Path(root_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.samples = []

        extensions = {".mp4", ".avi", ".mov", ".mkv"}

        for label_name, label_idx in [("real", 0), ("fake", 1)]:
            label_dir = self.root_dir / split / label_name
            if label_dir.exists():
                for f in label_dir.iterdir():
                    if f.suffix.lower() in extensions:
                        self.samples.append((str(f), label_idx))

    def _sample_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total <= 0:
            cap.release()
            raise RuntimeError(f"Empty video: {video_path}")

        indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
        frames = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif frames:
                frames.append(frames[-1].copy())

        cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1].copy())

        return frames[:self.num_frames]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._sample_frames(path)

        if self.transform:
            frames = [self.transform(image=f)["image"] for f in frames]

        return torch.stack(frames), label
