FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure unbuffered output for logs
ENV PYTHONUNBUFFERED=1

EXPOSE 7860

# Start OpenEnv-compatible API server for Spaces health checks (/reset, /step, /state)
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]