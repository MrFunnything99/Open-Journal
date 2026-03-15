FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (build tools etc. for native libs like sqlite-vec)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (prefer binaries, avoid resolver hang on networkx)
COPY requirements.txt .
ENV PIP_PREFER_BINARY=1 PIP_DEFAULT_TIMEOUT=120
RUN pip install -r requirements.txt

# Copy backend source
COPY . .

# Fly.io (and many PaaS) expect the app to listen on $PORT.
# Default to 8080 and expose it via uvicorn.
ENV PORT=8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

