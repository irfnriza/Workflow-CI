FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY MLProject/ /app/MLProject/

# Set environment variables
ENV MLFLOW_TRACKING_URI=file:./mlruns
ENV PYTHONUNBUFFERED=1

WORKDIR /app/MLProject

# Expose MLflow serving port
EXPOSE 5000

# Default command: serve the model
CMD ["mlflow", "models", "serve", "-m", "models/best_bilstm_model.h5", "-h", "0.0.0.0", "-p", "5000"]
