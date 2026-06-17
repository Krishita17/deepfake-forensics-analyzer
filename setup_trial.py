"""
Quick trial setup — creates a minimal 15-image dataset to test the full pipeline
without downloading any large datasets.

Two modes:
  1. --generate : Creates synthetic face-like images (works instantly, no download)
  2. --from-folder : Takes a folder of your own images and splits them into the project structure

Usage:
  python setup_trial.py --generate
  python setup_trial.py --from-folder /path/to/your/images --label real
"""

import os
import argparse
import shutil
import numpy as np
import cv2
from pathlib import Path


PROCESSED_DIR = Path("data/processed")

SPLITS = {
    "train": 9,
    "val": 3,
    "test": 3,
}


def create_dirs():
    for split in SPLITS:
        for label in ["real", "fake"]:
            (PROCESSED_DIR / split / label).mkdir(parents=True, exist_ok=True)
    print("[✓] Directory structure ready")


def generate_synthetic_face(index, is_fake=False, size=380):
    img = np.zeros((size, size, 3), dtype=np.uint8)

    skin_color = (200, 180, 160) if not is_fake else (210, 175, 155)
    cv2.ellipse(img, (size // 2, size // 2), (120, 160), 0, 0, 360, skin_color, -1)

    eye_y = size // 2 - 30
    cv2.circle(img, (size // 2 - 45, eye_y), 15, (255, 255, 255), -1)
    cv2.circle(img, (size // 2 + 45, eye_y), 15, (255, 255, 255), -1)
    cv2.circle(img, (size // 2 - 45, eye_y), 7, (80, 50, 30), -1)
    cv2.circle(img, (size // 2 + 45, eye_y), 7, (80, 50, 30), -1)

    nose_y = size // 2 + 10
    pts = np.array([[size // 2, nose_y - 20], [size // 2 - 12, nose_y + 10], [size // 2 + 12, nose_y + 10]])
    cv2.polylines(img, [pts], True, (170, 150, 130), 2)

    mouth_y = size // 2 + 50
    cv2.ellipse(img, (size // 2, mouth_y), (30, 10), 0, 0, 180, (150, 100, 100), 2)

    if is_fake:
        blend_y = size // 2
        cv2.line(img, (size // 2 - 130, blend_y - 50), (size // 2 - 130, blend_y + 80), (180, 160, 140), 3)
        noise = np.random.randint(0, 15, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        block_size = 8
        for y in range(0, size - block_size, block_size * 4):
            for x in range(0, size - block_size, block_size * 4):
                if np.random.random() > 0.7:
                    img[y:y+block_size, x:x+block_size] = cv2.GaussianBlur(
                        img[y:y+block_size, x:x+block_size], (3, 3), 0
                    )

    bg_noise = np.random.randint(0, 5, img.shape, dtype=np.uint8)
    img = cv2.add(img, bg_noise)

    return img


def generate_trial_dataset():
    create_dirs()

    total_per_label = sum(SPLITS.values())
    print(f"\nGenerating {total_per_label} real + {total_per_label} fake = {total_per_label * 2} images total\n")

    idx = 0
    for split, count in SPLITS.items():
        for i in range(count):
            real_img = generate_synthetic_face(idx, is_fake=False)
            fake_img = generate_synthetic_face(idx, is_fake=True)

            real_path = PROCESSED_DIR / split / "real" / f"real_{idx:04d}.png"
            fake_path = PROCESSED_DIR / split / "fake" / f"fake_{idx:04d}.png"

            cv2.imwrite(str(real_path), real_img)
            cv2.imwrite(str(fake_path), fake_img)

            print(f"  {split:5s} | real_{idx:04d}.png + fake_{idx:04d}.png")
            idx += 1

    print(f"\n[✓] Trial dataset created! Layout:\n")
    for split in SPLITS:
        for label in ["real", "fake"]:
            d = PROCESSED_DIR / split / label
            n = len(list(d.glob("*")))
            print(f"    data/processed/{split}/{label}/  →  {n} images")

    print(f"\n[✓] Total: {idx * 2} images across train/val/test")
    print("\nNext steps:")
    print("  1. Train:      python train.py")
    print("  2. Inference:   python inference.py data/processed/test/fake/fake_0012.png")
    print("  3. Dashboard:   streamlit run app.py")


def import_from_folder(folder_path, label):
    create_dirs()

    folder = Path(folder_path)
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([f for f in folder.iterdir() if f.suffix.lower() in extensions])

    if not images:
        print(f"[✗] No images found in {folder}")
        return

    total_needed = sum(SPLITS.values())
    selected = images[:total_needed]

    if len(selected) < total_needed:
        print(f"[!] Found only {len(selected)} images, need {total_needed}. Using what's available.")

    print(f"\nImporting {len(selected)} images as '{label}' into project structure:\n")

    idx = 0
    for split, count in SPLITS.items():
        for i in range(min(count, len(selected) - idx)):
            src = selected[idx]
            dst = PROCESSED_DIR / split / label / f"{label}_{idx:04d}{src.suffix}"
            shutil.copy2(src, dst)
            print(f"  {split:5s} | {src.name}  →  {dst.name}")
            idx += 1

    print(f"\n[✓] Imported {idx} images as '{label}'")
    print(f"\n[!] You still need '{('fake' if label == 'real' else 'real')}' images.")
    print(f"    Run again with: python setup_trial.py --from-folder /path/to/other/images --label {'fake' if label == 'real' else 'real'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup a minimal trial dataset (15 images)")
    parser.add_argument("--generate", action="store_true",
                        help="Generate synthetic face images for instant testing")
    parser.add_argument("--from-folder", type=str,
                        help="Import your own images from a folder")
    parser.add_argument("--label", type=str, choices=["real", "fake"], default="real",
                        help="Label for imported images (default: real)")
    args = parser.parse_args()

    if args.from_folder:
        import_from_folder(args.from_folder, args.label)
    elif args.generate:
        generate_trial_dataset()
    else:
        print("Quick Trial Setup for DeepFake Forensics Analyzer")
        print("=" * 50)
        print()
        print("Option 1 — Instant synthetic data (no downloads):")
        print("  python setup_trial.py --generate")
        print()
        print("Option 2 — Use your own images:")
        print("  python setup_trial.py --from-folder ~/my-real-faces --label real")
        print("  python setup_trial.py --from-folder ~/my-fake-faces --label fake")
        print()
        print("Option 3 — Download 15 from Kaggle 140K dataset:")
        print("  1. Go to: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
        print("  2. Click 'Download' but cancel after a few seconds")
        print("  3. Or use the Kaggle API to download just a sample:")
        print("     kaggle datasets download -d xhlulu/140k-real-and-fake-faces --unzip -p data/raw/")
        print("     Then pick 15 images manually into data/processed/")
