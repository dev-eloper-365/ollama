# Docker Setup Guide

## Quick Start

### 1. Build and Start Services

```bash
# Build images and start services
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Pull Required Models

```bash
# Pull embedding model
docker-compose exec ollama ollama pull embeddinggemma

# Pull generation model (optional)
docker-compose exec ollama ollama pull gemma3:12b
```

### 3. Test the API

```bash
# Health check
curl http://localhost:8080/health

# Test embeddings
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"]}'
```

## Services

- **Ollama**: `http://localhost:11434`
- **API Server**: `http://localhost:8080`

## Docker Compose Files

### Standard Setup
```bash
docker-compose up -d
```

### Development Mode (with hot-reload)
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

### Production Mode (with resource limits)
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

## Makefile Commands

```bash
# Build images
make build

# Start services
make up

# Start in development mode
make up-dev

# View logs
make logs

# Test API
make test

# Pull models
make pull-models

# Stop services
make down

# Clean everything
make clean

# Check status
make status
```

## Volumes

- `ollama_data`: Stores Ollama models
- `chroma_data`: Stores ChromaDB data

## Environment Variables

### API Service
- `PORT`: API server port (default: 8080)
- `OLLAMA_URL`: Ollama service URL (default: http://ollama:11434)
- `EMBED_MODEL`: Embedding model name (default: embeddinggemma)
- `CHROMA_PERSIST_DIR`: ChromaDB persistence directory

### Ollama Service
- `OLLAMA_HOST`: Host and port (default: 0.0.0.0:11434)

## Troubleshooting

### Check Logs
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs api
docker-compose logs ollama

# Follow logs
docker-compose logs -f
```

### Restart Services
```bash
# Restart all
docker-compose restart

# Restart specific service
docker-compose restart api
docker-compose restart ollama
```

### Rebuild After Code Changes
```bash
# Rebuild API service
docker-compose build api
docker-compose up -d api
```

### Access Container Shell
```bash
# API container
docker-compose exec api bash

# Ollama container
docker-compose exec ollama sh
```

### Check Health
```bash
# API health
curl http://localhost:8080/health

# Ollama health
curl http://localhost:11434/api/tags
```

## GPU Support

To enable GPU support for Ollama, uncomment the GPU section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

**Requirements:**
- NVIDIA Docker runtime installed
- nvidia-docker2 package

## Network Configuration

Services communicate via Docker's internal network:
- API service connects to Ollama at `http://ollama:11434`
- External access:
  - Ollama: `http://localhost:11434`
  - API: `http://localhost:8080`

## Data Persistence

- Models are stored in `ollama_data` volume
- ChromaDB data is stored in `chroma_data` volume
- Volumes persist even when containers are stopped

To remove all data:
```bash
docker-compose down -v
```

