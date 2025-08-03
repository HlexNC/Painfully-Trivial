# Multi-stage build for optimized production image
# Stage 1: Builder
FROM python:3.10-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download models during build (optional, can be done at runtime)
# RUN mkdir -p /app/models && \
#     wget -O /app/models/waste_detector_best.pt \
#     https://github.com/HlexNC/Painfully-Trivial/releases/download/v1.0.0/waste_detector_best.pt

# Stage 2: Production
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libgl1-mesa-glx \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxss1 \
    libasound2 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils \
    wget \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Security: Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/data /app/.streamlit && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=appuser:appuser . /app/

# Pre-download models (optional, can be done at runtime)
# COPY --from=builder /app/models /app/models

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8501", \
     "--server.baseUrlPath", "", \
     "--server.enableStaticServing", "true"]
