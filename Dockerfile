# Stage 1: Builder
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the source code, configuration, and frontend
COPY config ./config
COPY src ./src
COPY app.py .
COPY api.py .
COPY frontend ./frontend

# Set PYTHONPATH so that the app can find the src modules
ENV PYTHONPATH=/app
ENV PORT=8000

# Expose default port
EXPOSE 8000

# Run the FastAPI application using uvicorn (compatible with Render's dynamic $PORT)
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
