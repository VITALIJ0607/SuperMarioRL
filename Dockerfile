# Super Mario Bros RL Training mit PPO
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY super_mario_rl.py .
COPY README.md .
COPY IMPLEMENTATION_STEPS.md .

# Create directories for outputs
RUN mkdir -p models logs ppo_mario_tensorboard

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Expose TensorBoard port
EXPOSE 6006

# Default command: run training
CMD ["python", "super_mario_rl.py"]
