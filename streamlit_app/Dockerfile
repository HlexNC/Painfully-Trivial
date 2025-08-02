# Multi-stage build for optimized image size
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
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
    libglib2.0-0 \
    libgtk-3-0 \
    libnspr4 \
    libnss3 \
    libxss1 \
    libasound2 \
    fonts-liberation \
    libappindicator3-1 \
    xdg-utils \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models /app/.streamlit && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=appuser:appuser . /app/

# Create Streamlit config
RUN echo '\
[general]\n\
email = ""\n\
\n\
[server]\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
port = 8501\n\
\n\
[browser]\n\
serverAddress = "0.0.0.0"\n\
gatherUsageStats = false\n\
\n\
[theme]\n\
primaryColor = "#2E7D32"\n\
backgroundColor = "#FFFFFF"\n\
secondaryBackgroundColor = "#F0F2F6"\n\
textColor = "#262730"\n\
font = "sans serif"\n\
' > /app/.streamlit/config.toml

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]