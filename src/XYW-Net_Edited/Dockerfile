# Use official PyTorch image
FROM pytorch/pytorch:2.0-cuda11.8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libsm6 \
    libxext6 \
    libxrender-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Create directories
RUN mkdir -p /app/data /app/checkpoints /app/results /app/test_results /app/robustness_results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Expose port for TensorBoard
EXPOSE 6006

# Default command
CMD ["/bin/bash"]

# Example usage:
# docker build -t bio-xyw-net:latest .
# docker run --gpus all -it -v $(pwd)/data:/app/data bio-xyw-net:latest
# python bio_test.py --image test_image.jpg
