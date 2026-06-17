# DeepFake Forensics Analyzer

A comprehensive multi-modal deepfake detection system combining ensemble neural networks, frequency domain forensics, facial landmark analysis, and temporal consistency checking — with interactive visualization dashboards and detailed forensic reports.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Key Features

### Multi-Model Ensemble Architecture
- **EfficientNet-B4** with SRM (Steganalysis Rich Model) filters and high-pass frequency branching
- **XceptionNet** with depthwise separable convolutions and manipulation-type classification
- **Capsule Network** with dynamic routing for capturing spatial hierarchies in forged regions
- **Frequency-Aware Network** with DCT layer, FFT phase analysis, and spectral gating
- **Meta-classifier** fusion with learned model weighting and Monte Carlo uncertainty estimation

### Advanced Forensic Analysis
- **FFT / DCT / Wavelet** frequency decomposition with radial power spectrum slope analysis
- **Facial landmark consistency** — symmetry scoring, proportion deviation, jaw contour smoothness
- **Skin texture forensics** — Gabor filter uniformity, Laplacian texture variance, cross-region color consistency
- **Blending artifact detection** — boundary edge density, Lab color space discontinuity analysis
- **Temporal consistency** — optical flow smoothness, flickering detection, face tracking stability

### Visualization & Reporting
- Interactive **Streamlit dashboard** with Plotly gauge charts, radar plots, and frame-by-frame timelines
- Automated **HTML forensic reports** with score breakdowns and analysis figures
- Publication-quality **matplotlib figures** — FFT spectra, DCT energy maps, wavelet decomposition, anomaly indicators
- **Grad-CAM heatmaps** with multi-scale fusion for model explainability

## Project Structure

```
├── app.py                          # Streamlit web dashboard
├── train.py                        # Training pipeline with Focal Loss, AMP, early stopping
├── inference.py                    # CLI inference with full forensic analysis
├── download_datasets.py            # Dataset setup helper
├── configs/
│   └── default.yaml                # Full configuration (models, training, analysis)
├── src/
│   ├── models/
│   │   ├── efficientnet_detector.py   # EfficientNet + SRM + frequency branch
│   │   ├── xception_detector.py       # Xception with manipulation head
│   │   ├── capsule_network.py         # Capsule net with dynamic routing
│   │   ├── frequency_network.py       # DCT + FFT + spectral gating network
│   │   ├── ensemble.py                # Ensemble with meta-classifier + uncertainty
│   │   └── attention_module.py        # Multi-head attention, cross-modal, pyramid fusion
│   ├── analysis/
│   │   ├── frequency_analysis.py      # FFT, DCT, wavelet, spectral slope
│   │   ├── facial_forensics.py        # Landmarks, texture, blending artifacts
│   │   ├── temporal_analysis.py       # Optical flow, flickering, face tracking
│   │   └── grad_cam.py               # Grad-CAM + guided Grad-CAM + multi-scale
│   ├── preprocessing/
│   │   ├── face_extractor.py          # MTCNN face extraction for images/videos
│   │   ├── augmentations.py           # Training augmentations + JPEG compression
│   │   └── dataset.py                 # Image and video dataset loaders
│   ├── visualization/
│   │   ├── plotting.py                # Forensic visualization (frequency, ensemble, temporal)
│   │   └── report_generator.py        # HTML + JSON report generation
│   └── utils/
│       └── metrics.py                 # ROC, PR, EER, confusion matrix plotting
├── tests/
│   ├── test_models.py
│   └── test_analysis.py
├── configs/default.yaml
├── requirements.txt
└── setup.py
```

## Installation & Setup

### Prerequisites
- Python 3.9+
- CUDA-capable GPU (recommended, CPU works but slower)

### Step 1: Clone the repository
```bash
git clone https://github.com/Krishita17/deepfake-forensics-analyzer.git
cd deepfake-forensics-analyzer
```

### Step 2: Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Setup datasets
```bash
python download_datasets.py --setup-dirs
```

Then download one or more datasets (see below) and place extracted face images into:
```
data/processed/train/real/
data/processed/train/fake/
data/processed/val/real/
data/processed/val/fake/
```

## Datasets

| Dataset | Size | Description | Link |
|---------|------|-------------|------|
| **FaceForensics++** | ~1K videos | 4 manipulation methods (DF, F2F, FS, NT) | [GitHub](https://github.com/ondyari/FaceForensics) |
| **Celeb-DF v2** | 6,229 videos | High-quality celebrity deepfakes | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) |
| **DFDC** | 100K+ clips | Facebook Deepfake Detection Challenge | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) |
| **WildDeepfake** | 7,314 sequences | Real-world internet deepfakes | [GitHub](https://github.com/deepfakeinthewild/deepfake-in-the-wild) |
| **DeeperForensics** | 60K videos | Large-scale with quality perturbations | [GitHub](https://github.com/EndlessSora/DeeperForensics-1.0) |
| **140K Real/Fake Faces** | 140K images | Quick-start Flickr + StyleGAN | [Kaggle](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) |

## Usage

### Training
```bash
python train.py
```
Adjust hyperparameters in `configs/default.yaml`. Training uses Focal Loss, mixed-precision, cosine annealing, and early stopping.

### Inference (CLI)
```bash
# Analyze an image
python inference.py path/to/image.jpg

# Analyze a video
python inference.py path/to/video.mp4

# Skip report generation
python inference.py path/to/image.jpg --no-report
```

### Interactive Dashboard
```bash
streamlit run app.py
```
Open `http://localhost:8501` — upload images/videos and get real-time forensic analysis with interactive charts.

### Run Tests
```bash
pytest tests/ -v
```

## How It Works

### Detection Pipeline
1. **Face Extraction** — MTCNN detects and crops faces with configurable margins
2. **Multi-Model Inference** — Four specialized networks each produce predictions
3. **Frequency Analysis** — FFT, DCT, and wavelet decomposition reveal manipulation artifacts invisible to the eye
4. **Facial Forensics** — Landmark geometry, skin texture, and blending boundaries are analyzed
5. **Temporal Analysis** (video) — Optical flow consistency, flickering patterns, and face tracking smoothness
6. **Ensemble Fusion** — Meta-classifier combines all signals with learned weights + uncertainty estimation
7. **Visualization** — Scores, charts, heatmaps, and forensic reports are generated

### Scoring
The final forgery score is a weighted combination:
- **Image**: 45% Neural Network + 30% Frequency + 25% Facial Forensics
- **Video**: 40% Neural Network + 25% Frequency + 35% Temporal Consistency

## License

MIT

## Author

**Krishita17** — [GitHub](https://github.com/Krishita17)
