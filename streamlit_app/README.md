# 🗑️ Deggendorf Waste Sorting Assistant - Web Application

[![Docker Image](https://img.shields.io/badge/Docker-ghcr.io%2Fhlexnc%2Fwaste--sorting--assistant-blue?style=for-the-badge&logo=docker)](https://github.com/HlexNC/Painfully-Trivial/pkgs/container/waste-sorting-assistant)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31.0-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)

A professional web application for real-time waste bin detection, helping international students and residents in Deggendorf properly sort their waste using AI-powered computer vision.

## 🚀 Quick Start

### Option 1: Run with Docker (Recommended)

```bash
# Pull and run the latest image
docker pull ghcr.io/hlexnc/waste-sorting-assistant:latest
docker run -p 8501:8501 ghcr.io/hlexnc/waste-sorting-assistant:latest
```

Open http://localhost:8501 in your browser.

### Option 2: Run with Docker Compose

```bash
# Clone the repository
git clone https://github.com/HlexNC/Painfully-Trivial.git
cd Painfully-Trivial/streamlit_app

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f
```

### Option 3: Run Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 🎯 Features

### 1. **Live Detection** 📸
- Real-time webcam detection
- Image upload support
- Adjustable confidence threshold
- Multi-language disposal guidelines

### 2. **Model Training Interface** 🔧
- Configure training parameters
- Data augmentation settings
- Real-time training progress
- Performance visualization

### 3. **Performance Analytics** 📊
- Detailed metrics (mAP, Precision, Recall)
- Per-class performance breakdown
- Confusion matrix visualization
- Inference speed benchmarks

### 4. **Professional UI** 🎨
- Mobile-responsive design
- Dark mode support
- Intuitive navigation
- Clean, modern interface

## 🐳 Docker Deployment

### Building the Image

```bash
# Build for multiple platforms
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ghcr.io/hlexnc/waste-sorting-assistant:latest \
  --push .
```

### Environment Variables

```bash
# Create .env file
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_HEADLESS=true
MODEL_DOWNLOAD_URL=https://github.com/HlexNC/Painfully-Trivial/releases/download/v1.0.0/waste_detector_best.pt
```

### Production Deployment

```bash
# Run with docker-compose (includes monitoring)
docker-compose --profile production --profile monitoring up -d
```

## 📱 Camera Access

### Desktop Browser
- Chrome/Edge: Allow camera permissions when prompted
- Firefox: Click the camera icon in the address bar

### Mobile Browser
- Ensure HTTPS connection (required for camera access)
- Grant camera permissions in browser settings

### Troubleshooting Camera Issues
1. Check browser permissions
2. Ensure no other app is using the camera
3. Try refreshing the page
4. Use Chrome/Edge for best compatibility

## 🔧 Configuration

### Model Configuration

The app automatically downloads the trained model from GitHub releases. To use a custom model:

1. Place your model in the `models/` directory
2. Update the `MODEL_PATH` in `app.py`
3. Restart the application

### Streamlit Configuration

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#2E7D32"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"

[server]
maxUploadSize = 200
enableXsrfProtection = false
```

## 🚀 CI/CD Pipeline

The project uses GitHub Actions for automated deployment:

1. **Push to main** → Build Docker image
2. **Create release** → Tag with version
3. **Security scan** → Trivy vulnerability scanning
4. **Deploy to GHCR** → Available at `ghcr.io`

### Manual Deployment

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Build and push
docker buildx build --push \
  --platform linux/amd64,linux/arm64 \
  -t ghcr.io/hlexnc/waste-sorting-assistant:latest .
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@0.5 | 95.2% |
| mAP@0.5:0.95 | 78.4% |
| Precision | 92.8% |
| Recall | 89.6% |
| FPS (RTX 3090) | 156 |
| FPS (CPU) | 12 |

## 🛡️ Security

- Non-root container user
- Health checks enabled
- Trivy vulnerability scanning
- No sensitive data in images
- Environment variable configuration

## 📈 Monitoring (Optional)

### Prometheus Metrics

Access at http://localhost:9090

- Request latency
- Model inference time
- Memory usage
- Error rates

### Grafana Dashboards

Access at http://localhost:3000 (admin/admin)

- Real-time performance metrics
- Historical trends
- Alert configuration

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application |
| `/_stcore/health` | GET | Health check |
| `/_stcore/stream` | WSS | WebSocket stream |
| `/media/*` | GET | Static media files |

## 🐛 Troubleshooting

### Common Issues

1. **Model download fails**
   ```bash
   # Manually download model
   wget https://github.com/HlexNC/Painfully-Trivial/releases/download/v1.0.0/waste_detector_best.pt
   mkdir -p models && mv waste_detector_best.pt models/
   ```

2. **Camera not working**
   - Ensure HTTPS connection
   - Check browser permissions
   - Try different browser

3. **High memory usage**
   ```bash
   # Limit container memory
   docker run -m 2g ghcr.io/hlexnc/waste-sorting-assistant:latest
   ```

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/HlexNC/Painfully-Trivial/issues)
- **Discussions**: [GitHub Discussions](https://github.com/HlexNC/Painfully-Trivial/discussions)
- **Email**: hlex@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- TH Deggendorf for academic support
- Prof. Dr. Glauner for guidance
- Ultralytics for YOLOv8
- Streamlit for the amazing framework

---

Made with ❤️ by Sameer, Fares, and Alex at TH Deggendorf