"""
Dataset download helper.

Supported datasets and instructions:

1. FaceForensics++ (FF++)
   - URL: https://github.com/ondyari/FaceForensics
   - Request access via Google Form on their GitHub
   - Contains: Deepfakes, Face2Face, FaceSwap, NeuralTextures
   - ~1000 original videos + manipulated versions

2. Celeb-DF v2
   - URL: https://github.com/yuezunli/celeb-deepfakeforensics
   - Request access via form
   - Contains: 590 real + 5639 synthesized deepfake videos

3. DFDC (Deepfake Detection Challenge)
   - URL: https://www.kaggle.com/c/deepfake-detection-challenge/data
   - Requires Kaggle account
   - Contains: 100,000+ clips (train ~470GB)

4. WildDeepfake
   - URL: https://github.com/deepfakeinthewild/deepfake-in-the-wild
   - Real-world deepfakes from the internet
   - 7,314 face sequences from 707 deepfake videos

5. DeeperForensics-1.0
   - URL: https://github.com/EndlessSora/DeeperForensics-1.0
   - 60,000 videos, 17.6M frames
   - High quality perturbations

After downloading, organize data as:
    data/processed/
    ├── train/
    │   ├── real/
    │   └── fake/
    ├── val/
    │   ├── real/
    │   └── fake/
    └── test/
        ├── real/
        └── fake/
"""

import os
import argparse
from pathlib import Path


def setup_directory_structure():
    dirs = [
        "data/raw",
        "data/processed/train/real",
        "data/processed/train/fake",
        "data/processed/val/real",
        "data/processed/val/fake",
        "data/processed/test/real",
        "data/processed/test/fake",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("Directory structure created.")


def download_sample_data():
    try:
        import gdown
    except ImportError:
        print("Install gdown: pip install gdown")
        return

    print("Note: Full datasets require manual download due to access restrictions.")
    print("See docstring above for dataset URLs and access instructions.")
    print("\nFor quick testing, you can use:")
    print("  - FaceForensics++ sample: https://github.com/ondyari/FaceForensics")
    print("  - 140k Real and Fake Faces (Kaggle): https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces")
    print("  - Deepfake Face (Kaggle): https://www.kaggle.com/datasets/dagnelies/deepfake-faces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup datasets for DeepFake detection")
    parser.add_argument("--setup-dirs", action="store_true", help="Create directory structure")
    parser.add_argument("--download-sample", action="store_true", help="Show download instructions")
    args = parser.parse_args()

    if args.setup_dirs:
        setup_directory_structure()
    if args.download_sample:
        download_sample_data()
    if not args.setup_dirs and not args.download_sample:
        setup_directory_structure()
        download_sample_data()
