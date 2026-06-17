from setuptools import setup, find_packages

setup(
    name="deepfake-forensics-analyzer",
    version="1.0.0",
    author="Krishita17",
    description="Multi-modal deepfake detection with frequency analysis, facial forensics, and ensemble neural networks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
    ],
)
