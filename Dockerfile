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

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_BASE_URL=${API_BASE_URL:-https://api.openai.com/v1}
ENV MODEL_NAME=${MODEL_NAME:-gpt-4}
ENV HF_TOKEN=${HF_TOKEN:-}

EXPOSE 7860

# Start OpenEnv-compatible API server for Spaces health checks (/reset, /step, /state)
CMD ["python", "-m", "server.app"]