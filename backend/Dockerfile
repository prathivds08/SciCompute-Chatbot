# Use official slim Python runtime
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install system dependencies required for building C extensions (e.g., Kùzu, NumPy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Layer cache: Copy requirements first and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy the backend code into /app/backend to preserve the "backend" package namespace
# for internal imports (e.g., "from backend.config import settings")
COPY . /app/backend

# Create directory for persistent runtime data (databases, sandbox scripts)
RUN mkdir -p /app/backend/data/sandbox

EXPOSE 8000

# Start FastAPI application with uvicorn
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
