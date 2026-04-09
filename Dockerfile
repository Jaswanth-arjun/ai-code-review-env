FROM python:3.10-slim

WORKDIR /app

# Pure-Python deps use wheels; skip build-essential to keep image small and builds fast on 2 vCPU.

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