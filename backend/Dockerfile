# Multi-stage: Stage 1 build Vite frontend, Stage 2 serve FastAPI + static
# Build from repo root: docker build -f backend/Dockerfile .

# --- Stage 1: Node - build Vite frontend ---
FROM node:20-alpine AS frontend-build
WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

# Copy frontend source (backend excluded via .dockerignore when building from repo root)
COPY . .
RUN npm run build

# --- Stage 2: Python - FastAPI + static frontend ---
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for native libs (sqlite-vec)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
ENV PIP_PREFER_BINARY=1 PIP_DEFAULT_TIMEOUT=120
RUN pip install -r requirements.txt

# Backend source (build context is repo root)
COPY backend/ .

# Frontend build from stage 1 → static
COPY --from=frontend-build /app/dist ./static

ENV PORT=8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
