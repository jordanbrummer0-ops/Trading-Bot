# Trading Bot Dockerfile
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash tradingbot

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/tradingbot/.local

# Copy application code
COPY --chown=tradingbot:tradingbot . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/analysis_charts \
    && chown -R tradingbot:tradingbot /app

# Switch to non-root user
USER tradingbot

# Add local Python packages to PATH
ENV PATH=/home/tradingbot/.local/bin:$PATH

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV TZ=UTC

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501/healthz', timeout=5)" || exit 1

# Expose ports
EXPOSE 8501 8000

# Default command (can be overridden)
CMD ["python", "main.py"]