<h1 align="center">
  <img src="./docs/img/painfully-trivial-banner.png" alt="Painfully Trivial - ML Projects Banner" width="800"/>
</h1>

<h3 align="center">🚀 Advanced Machine Learning & Computer Vision Solutions</h3>
<h3 align="center">Deutsche Bahn Delay Prediction | Waste Sorting Assistant</h3>

<p align="center">
  <a href="https://github.com/HlexNC/Painfully-Trivial">
    <img src="https://img.shields.io/badge/GitHub-Painfully_Trivial-181717?style=for-the-badge&logo=github" alt="GitHub Badge">
  </a>
  <a href="https://www.python.org/downloads/release/python-3100/">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="MIT License">
  </a>
</p>

<p align="center">
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker Ready">
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-Demo-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit Demo">
  </a>
  <a href="https://github.com/features/actions">
    <img src="https://img.shields.io/badge/CI/CD-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white" alt="GitHub Actions">
  </a>
</p>

<p align="center">
  <a href="#-live-demo">Live Demo</a> •
  <a href="#-features">Features</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-projects">Projects</a> •
  <a href="#-deployment">Deployment</a> •
  <a href="#-contributing">Contributing</a>
</p>

---

## 🎯 Overview

**Painfully Trivial** showcases two cutting-edge machine learning applications developed as part of our academic journey at TH Deggendorf. These projects demonstrate practical applications of ML/DL in solving real-world problems.

> [!NOTE]
> Both projects were developed by Sameer, Fares, and Alex as part of the Machine Learning course and have been successfully graded by the university.

## 🌟 Live Demo

<p align="center">
  <a href="https://painfully-trivial.streamlit.app">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Waste_Sorting_Assistant-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
</p>

> [!TIP]
> The demo supports real-time camera input! Allow camera permissions when prompted for the best experience.

## 📋 Features

### 🗑️ Waste Sorting Assistant (Computer Vision)
- **Real-time Detection**: Identifies German waste bins using YOLOv8
- **4 Categories**: Biomüll, Glas, Papier, Restmüll
- **Live Camera Support**: Works with webcam or phone camera
- **Multi-language Instructions**: Guides proper waste disposal
- **Custom Dataset**: 466 locally captured images from Deggendorf

### 🚄 Deutsche Bahn Delay Predictor (ML Regression)
- **92.4% Accuracy**: Outperforms baseline predictions
- **2M+ Records**: Trained on extensive Deutsche Bahn dataset
- **Feature Engineering**: Temporal patterns, station characteristics
- **Multiple Models**: Linear Regression, KNN, Random Forest
- **Production Ready**: Includes preprocessing pipeline and deployment code

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10 or higher
python --version

# CUDA-capable GPU (optional, for faster training)
nvidia-smi
```

### Installation

```bash
# Clone the repository
git clone https://github.com/HlexNC/Painfully-Trivial.git
cd painfully-trivial

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> [!IMPORTANT]
> For GPU support, install PyTorch with CUDA:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

## 📂 Project Structure

```
painfully-trivial/
├── 📁 cv_garbage/                 # Computer Vision Project
│   ├── 📁 models/                 # Trained YOLO models
│   ├── 📁 YOLO_Dataset/          # Annotated dataset
│   ├── 📁 raw_images/            # Original photos
│   ├── 📄 2-Computer-Vision.py   # Main CV notebook
│   └── 📄 YOLO_Model.py         # Model training scripts
├── 📁 ml_deutsche_bahn/          # ML Regression Project
│   ├── 📄 1-Machine-Learning-v2.py
│   └── 📁 data/
├── 📁 streamlit_app/             # Streamlit Demo App
│   ├── 📄 app.py
│   ├── 📄 pages/
│   └── 📄 utils/
├── 📁 .github/workflows/         # CI/CD Pipeline
├── 📄 Dockerfile
├── 📄 requirements.txt
└── 📄 README.md
```

## 🔬 Projects

### 1. Waste Sorting Assistant (Computer Vision)

<details>
<summary><b>🗑️ Click to expand project details</b></summary>

#### Problem Statement
International students in Deggendorf struggle with the German waste sorting system due to language barriers and unfamiliar color coding.

#### Solution
A real-time object detection system that:
- Identifies waste bin types from camera input
- Provides disposal guidelines in multiple languages
- Works on mobile devices and laptops

#### Technical Details
- **Model**: YOLOv8 (fine-tuned)
- **Dataset**: 466 custom images
- **Classes**: 4 (Biomüll, Glas, Papier, Restmüll)
- **Training**: 50 epochs, batch size 4
- **Performance**: 95%+ mAP@0.5

#### Usage
```python
from ultralytics import YOLO
import cv2

# Load model
model = YOLO('cv_garbage/models/waste_detector_best.pt')

# Run inference
results = model.predict(source=0, conf=0.65, show=True)  # 0 for webcam
```

</details>

### 2. Deutsche Bahn Delay Predictor

<details>
<summary><b>🚄 Click to expand project details</b></summary>

#### Problem Statement
Predict train arrival delays to help passengers plan better and assist DB in operational optimization.

#### Solution
A supervised regression model that:
- Predicts delays in minutes
- Uses historical patterns and station data
- Achieves 92.4% improvement over baseline

#### Technical Details
- **Dataset**: 2,061,357 records
- **Features**: 16 engineered features
- **Models**: Linear Regression, KNN, Random Forest
- **Best Model**: Random Forest (MSE: 0.8791)
- **Validation**: 60-20-20 train-val-test split

#### Key Insights
- Departure delay is the strongest predictor
- Rush hours show higher delays
- Station-specific patterns matter

</details>

## 🌐 Streamlit Deployment

### Local Development

```bash
cd streamlit_app
streamlit run app.py
```

### Docker Deployment

```bash
# Build image
docker build -t painfully-trivial:latest .

# Run container
docker run -p 8501:8501 painfully-trivial:latest
```

> [!TIP]
> For camera access in Docker, add `--device=/dev/video0` on Linux or use appropriate flags for your OS.

### CI/CD Pipeline

The project includes automated deployment to GitHub Container Registry:

```yaml
# .github/workflows/deploy.yml
name: Build and Deploy

on:
  push:
    branches: [main]
  release:
    types: [created]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Login to GHCR
        uses: docker/login-action@v2
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
```

## 🔧 Configuration

### Model Management

> [!NOTE]
> Models are stored in GitHub Releases due to size constraints. The app automatically downloads them on first run.

To use custom models:

1. **Via GitHub Release**:
   ```bash
   # Upload your model to a new release
   gh release create v1.0.0 ./models/your_model.pt
   ```

2. **Via Environment Variable**:
   ```bash
   export MODEL_PATH="https://your-storage.com/model.pt"
   ```

### Retraining Models

The Streamlit app includes a retraining interface:

1. Navigate to "Model Training" page
2. Upload new labeled data
3. Configure hyperparameters
4. Monitor training progress
5. Evaluate and deploy new model

## 📊 Performance Metrics

### Waste Sorting Assistant
- **Inference Speed**: 19.5 FPS on GPU
- **Accuracy**: 95%+ mAP@0.5
- **Model Size**: 22.5 MB

### DB Delay Predictor
- **Test MSE**: 0.8791
- **RMSE**: 0.938 minutes
- **Improvement**: 92.4% over baseline

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

```bash
# Fork the repo
# Create your feature branch
git checkout -b feature/AmazingFeature

# Commit your changes
git commit -m 'Add some AmazingFeature'

# Push to the branch
git push origin feature/AmazingFeature

# Open a Pull Request
```

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@misc{painfully-trivial2025,
  author = {Sameer, Fares, Alex},
  title = {Painfully Trivial: ML Solutions for Real-World Problems},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/HlexNC/Painfully-Trivial}}
}
```

## 👥 Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/TheSameerCode">
        <img src="https://github.com/TheSameerCode.png" width="100px;" alt="Sameer"/><br />
        <sub><b>Sameer</b></sub>
      </a><br />
      💻 📊 🔬
    </td>
    <td align="center">
      <a href="https://github.com/FaresM7">
        <img src="https://github.com/FaresM7.png" width="100px;" alt="Fares"/><br />
        <sub><b>Fares</b></sub>
      </a><br />
      💻 🎨 📱
    </td>
    <td align="center">
      <a href="https://github.com/HlexNC">
        <img src="https://github.com/HlexNC.png" width="100px;" alt="Alex"/><br />
        <sub><b>Alex</b></sub>
      </a><br />
      💻 🚀 📝
    </td>
  </tr>
</table>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TH Deggendorf** - For academic support and resources
- **Prof. Dr. Mayer** - Machine Learning course instructor
- **Prof. Dr. Glauner** - Computer Vision course instructor
- **Deutsche Bahn** - For providing the delays dataset
- **City of Deggendorf** - For allowing data collection

---

<p align="center">
  Made with ❤️ at TH Deggendorf
</p>

<p align="center">
  <a href="#-overview">Back to top ↑</a>
</p>
