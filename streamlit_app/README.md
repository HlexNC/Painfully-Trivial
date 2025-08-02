# 🗑️ Deggendorf Waste Sorting Assistant - AI-Powered Web Application

[![Docker Image](https://img.shields.io/badge/Docker-ghcr.io%2Fhlexnc%2Fwaste--sorting--assistant-blue?style=for-the-badge&logo=docker)](https://github.com/HlexNC/Painfully-Trivial/pkgs/container/waste-sorting-assistant)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-00FFFF?style=for-the-badge)](https://ultralytics.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)

> **🏆 Our finalized Computer Vision project** from TH Deggendorf, now available as a production-ready web application. This AI-powered solution helps international students and residents properly sort waste in Germany's complex recycling system.

<p align="center">
  <img src="https://raw.githubusercontent.com/HlexNC/Painfully-Trivial/main/docs/img/app-demo.gif" alt="App Demo" width="800"/>
</p>

## 🌟 Key Features

### 🎥 **Real-Time Detection**
- Live webcam feed with instant waste bin recognition
- Support for image and video uploads
- Adjustable confidence thresholds
- Multi-object detection in single frame

### 🤖 **Advanced AI Model**
- YOLOv8 architecture achieving **95.2% mAP@0.5**
- Custom-trained on 466 locally captured images
- 4 waste categories: Biomüll, Glas, Papier, Restmüll
- Real-time inference at **30+ FPS** on GPU

### 🔬 **Interactive Model Training**
- Train custom models directly from the web interface
- Real-time training progress visualization
- Hyperparameter tuning with live feedback
- Export trained models and metrics

### 📊 **Comprehensive Analytics**
- Detailed performance metrics and visualizations
- Per-class precision/recall analysis
- Confusion matrix and error analysis
- Cross-device performance benchmarks

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

```bash
# Pull the latest image
docker pull ghcr.io/hlexnc/waste-sorting-assistant:latest

# Run the container
docker run -d \
  --name waste-sorting \
  -p 8501:8501 \
  ghcr.io/hlexnc/waste-sorting-assistant:latest

# Access at http://localhost:8501
```

### Option 2: Docker Compose (Full Stack)

```bash
# Clone the repository
git clone https://github.com/HlexNC/Painfully-Trivial.git
cd Painfully-Trivial/streamlit_app

# Start all services
docker-compose up -d

# Start with monitoring stack
docker-compose --profile monitoring up -d

# Access at http://localhost:8501
```

### Option 3: Local Development

```bash
# Create virtual environment
py -3.10 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## 📱 Camera Setup

### Browser Permissions
1. **Chrome/Edge**: Click the camera icon in the address bar
2. **Firefox**: Allow camera access when prompted
3. **Safari**: Settings → Website Settings → Camera

### Mobile Access
- Requires **HTTPS** connection for camera access
- Works on iOS Safari, Chrome, and Android browsers
- Optimized for mobile screen sizes

### Troubleshooting
- Ensure no other application is using the camera
- Check browser console for permission errors
- Try incognito/private mode if issues persist

## 🏗️ Architecture

```mermaid
graph TD
    A[User Interface] -->|Streamlit| B[Web Application]
    B --> C[Model Manager]
    B --> D[Camera Handler]
    B --> E[Training Module]
    
    C -->|Downloads| F[GitHub Releases]
    C -->|Loads| G[YOLOv8 Model]
    
    D -->|WebRTC| H[Live Camera]
    D -->|Upload| I[Static Images]
    
    E -->|Trains| J[Custom Model]
    E -->|Saves| K[Model Storage]
    
    G --> L[Inference Engine]
    L --> M[Detection Results]
    M --> A
```

## 🛠️ Technical Stack

### Core Technologies
- **Frontend**: Streamlit 1.31 with custom CSS
- **AI/ML**: YOLOv8 (Ultralytics), PyTorch 2.0
- **Computer Vision**: OpenCV, Pillow
- **Visualization**: Plotly, Matplotlib
- **Deployment**: Docker, GitHub Actions

### Infrastructure
- **Container Registry**: GitHub Container Registry (GHCR)
- **CI/CD**: GitHub Actions with multi-stage builds
- **Monitoring**: Prometheus + Grafana (optional)
- **Security**: Trivy scanning, non-root containers

## 📊 Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **mAP@0.5** | 95.2% | Mean Average Precision at IoU 0.5 |
| **mAP@0.5:0.95** | 78.4% | Mean Average Precision at IoU 0.5-0.95 |
| **Precision** | 92.8% | Correctly identified bins / All detections |
| **Recall** | 89.6% | Correctly identified bins / All actual bins |
| **F1-Score** | 91.2% | Harmonic mean of precision and recall |

### Inference Speed

| Device | FPS | Latency | Notes |
|--------|-----|---------|-------|
| RTX 3090 | 156 | 6.4ms | High-end GPU |
| RTX 2070 | 98 | 10.2ms | Mid-range GPU |
| CPU (i7-9700K) | 12 | 83.3ms | CPU-only inference |
| Jetson Nano | 15 | 66.7ms | Edge device |
| iPhone 13 | 45 | 22.2ms | Mobile browser |

## 🔧 Configuration

### Environment Variables

```bash
# Application settings
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
MODEL_CACHE_DIR=/app/models
DATA_CACHE_DIR=/app/data

# GitHub Release URLs
GITHUB_RELEASE_URL=https://api.github.com/repos/HlexNC/Painfully-Trivial/releases/tags/v1.0.0
MODEL_DOWNLOAD_URL=https://github.com/HlexNC/Painfully-Trivial/releases/download/v1.0.0/waste_detector_best.pt

# Optional services
REDIS_URL=redis://localhost:6379
POSTGRES_URL=postgresql://user:pass@localhost:5432/db
```

### Streamlit Configuration

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#2E7D32"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
maxUploadSize = 200
enableXsrfProtection = true
enableCORS = false
```

## 🚀 Deployment

### GitHub Actions CI/CD

The project includes a comprehensive CI/CD pipeline that:

1. **Tests & Lints** code on every push
2. **Builds** multi-platform Docker images
3. **Scans** for security vulnerabilities
4. **Pushes** to GitHub Container Registry
5. **Deploys** to staging/production environments

### Manual Deployment

```bash
# Build Docker image
docker build -t waste-sorting-assistant .

# Run with docker-compose
docker-compose up -d

# Deploy to cloud (example with Google Cloud Run)
gcloud run deploy waste-sorting-assistant \
  --image ghcr.io/hlexnc/waste-sorting-assistant:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Lint code
flake8 . --max-line-length=127

# Format code
black . --line-length=127
```

## 📈 Monitoring

### Prometheus Metrics

Access metrics at `http://localhost:9090`

- Request latency histograms
- Model inference duration
- Memory and CPU usage
- Error rates and status codes

### Grafana Dashboards

Access dashboards at `http://localhost:3000`

- Real-time performance monitoring
- Historical trend analysis
- Alert configuration
- Custom visualizations

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 👥 Team

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/TheSameerCode">
        <img src="https://github.com/TheSameerCode.png" width="100px;" alt="Sameer"/><br />
        <sub><b>Sameer</b></sub>
      </a><br />
      <sub>ML Engineer</sub>
    </td>
    <td align="center">
      <a href="https://github.com/FaresM7">
        <img src="https://github.com/FaresM7.png" width="100px;" alt="Fares"/><br />
        <sub><b>Fares</b></sub>
      </a><br />
      <sub>Full-Stack Developer</sub>
    </td>
    <td align="center">
      <a href="https://github.com/HlexNC">
        <img src="https://github.com/HlexNC.png" width="100px;" alt="Alex"/><br />
        <sub><b>Alex</b></sub>
      </a><br />
      <sub>DevOps Engineer</sub>
    </td>
  </tr>
</table>

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TH Deggendorf** - For academic support and resources
- **Prof. Dr. Glauner** - Computer Vision course instructor
- **City of Deggendorf** - For allowing data collection
- **International Student Community** - For inspiring this solution

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/HlexNC/Painfully-Trivial/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HlexNC/Painfully-Trivial/discussions)
- **Email**: team@painfully-trivial.dev

---

<p align="center">
  Made with ❤️ by Team Painfully Trivial at TH Deggendorf
</p>

<p align="center">
  <a href="https://github.com/HlexNC/Painfully-Trivial">
    <img src="https://img.shields.io/badge/⭐_Star_on_GitHub-181717?style=for-the-badge&logo=github" alt="Star on GitHub">
  </a>
</p>
