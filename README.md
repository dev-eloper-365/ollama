# Ollama + ChromaDB Integration

A complete implementation for running Ollama locally and integrating it with ChromaDB for vector storage and retrieval.

## Features

- ✅ Generate embeddings using Ollama's local embedding models
- ✅ Store embeddings in ChromaDB for efficient vector search
- ✅ Query documents using semantic search
- ✅ Bulk ingestion from folders of documents
- ✅ Text generation using Ollama's language models
- ✅ Docker support for easy deployment

## Prerequisites

1. **Python Version**: Python 3.11 or 3.12 recommended (Python 3.14 has compatibility issues with ChromaDB)
2. **Install Ollama**: [Download and install Ollama](https://ollama.com) for your OS
3. **Pull required models**:
   ```bash
   # Pull the embedding model (recommended)
   ollama pull embeddinggemma
   
   # Pull the generation model (optional, for text generation)
   ollama pull gemma3:12b
   ```

## Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama Service

```bash
# Start Ollama server (foreground, for testing)
ollama serve

# OR run as background service (see systemd setup below)
```

Ollama will be available at `http://localhost:11434` by default.

### 3. Run the Example Script

```bash
python ollama_chroma.py
```

This will:
- Add sample documents to ChromaDB
- Query the database for similar documents
- Demonstrate text generation

## Usage

### Basic Integration

```python
from ollama_chroma import OllamaChromaIntegration

# Initialize
integration = OllamaChromaIntegration(
    embed_model="embeddinggemma",
    collection_name="my_docs"
)

# Add documents
texts = [
    "Llamas are members of the camelid family.",
    "ChromaDB is a lightweight vector DB."
]
integration.add_documents(texts)

# Query
results = integration.query("What is a llama?", n_results=3)
print(results)
```

### Bulk Ingestion

Process all documents from a folder:

```bash
python bulk_ingest.py /path/to/documents --collection my_docs
```

Options:
- `--collection`: ChromaDB collection name (default: `bulk_docs`)
- `--chunk-size`: Size of text chunks in characters (default: 1000)
- `--overlap`: Overlap between chunks (default: 200)
- `--extensions`: File extensions to process (default: `.txt .md .pdf`)
- `--batch-size`: Documents per batch (default: 10)

Example:
```bash
python bulk_ingest.py ./documents --collection research_papers --chunk-size 2000
```

## API Reference

### OllamaChromaIntegration

#### `__init__(ollama_url, embed_model, collection_name, persist_directory)`
Initialize the integration.

#### `get_embeddings(texts: List[str]) -> List[List[float]]`
Get embeddings from Ollama for a list of texts.

#### `add_documents(texts, ids=None, metadatas=None)`
Add documents to ChromaDB with embeddings from Ollama.

#### `query(query_text, n_results=3, return_documents=True) -> dict`
Query ChromaDB for similar documents.

#### `generate_text(prompt, model="gemma3:12b") -> str`
Generate text using Ollama's language models.

## Docker Deployment

### Using Docker Compose

**Quick Start:**
```bash
# Build and start all services
docker-compose up -d

# Pull required models
make pull-models
# or manually:
docker-compose exec ollama ollama pull embeddinggemma
docker-compose exec ollama ollama pull gemma3:12b

# Test the API
make test
# or manually:
curl http://localhost:8080/health
```

**Services:**
- **Ollama**: Running on port `11434`
- **API Server**: Running on port `8080`
- **ChromaDB**: Persisted in Docker volume `chroma_data`

**Development Mode:**
```bash
# Start with hot-reload for code changes
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

**Useful Commands:**
```bash
# View logs
make logs
# or
docker-compose logs -f

# Restart services
make restart

# Stop services
make down

# Clean everything (including volumes)
make clean
```

### Manual Docker Commands

```bash
# Build the API server image
docker build -t ollama-api .

# Run Ollama
docker run -d -v ollama_data:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

# Run API server
docker run -d \
  -p 8080:8080 \
  -v chroma_data:/app/chroma_db \
  -e OLLAMA_URL=http://host.docker.internal:11434 \
  --name ollama-api \
  ollama-api
```

### Manual Docker Run

```bash
docker run -d -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
```

## Systemd Service Setup (Linux)

1. Create service file:
```bash
sudo nano /etc/systemd/system/ollama.service
```

2. Add the following:
```ini
[Unit]
Description=Ollama Service
After=network.target

[Service]
Type=simple
User=your-username
ExecStart=/usr/bin/ollama serve
Restart=always
Environment="OLLAMA_HOST=0.0.0.0:11434"

[Install]
WantedBy=multi-user.target
```

3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable ollama
sudo systemctl start ollama
```

## Configuration

### Environment Variables

- `OLLAMA_HOST`: Host and port (default: `localhost:11434`)
- `OLLAMA_KV_CACHE_TYPE`: KV cache type for memory management

### Model Selection

**For Embeddings:**
- `embeddinggemma` (recommended)
- Other embedding models available in Ollama

**For Text Generation:**
- `gemma3:12b` (12B parameters, requires ~10-24GB VRAM)
- Quantized variants available (Q4_K_M, etc.)

## Hardware Requirements

### Memory Usage

- `gemma3:12b`: ~10-24GB VRAM depending on:
  - Quantization level
  - Context window size
  - KV cache settings

### Optimization Tips

1. **Use quantized models** for lower memory usage:
   ```bash
   ollama pull gemma3:12b:q4_0
   ```

2. **Reduce context window** if you get OOM errors

3. **Use dedicated embedding models** (smaller, faster) instead of generation models for embeddings

4. **Adjust KV cache**:
   ```bash
   export OLLAMA_KV_CACHE_TYPE=mmap  # or other options
   ```

## Troubleshooting

### Python Version Compatibility

**Note**: If you're using Python 3.14, you may encounter compatibility issues with ChromaDB due to pydantic 2.x changes. The script includes workarounds, but for best results, use Python 3.11 or 3.12.

### Ollama Connection Issues

```bash
# Test if Ollama is running
curl http://localhost:11434/api/tags

# Check Ollama status
ollama list
```

### Embedding Response Format

If you get unexpected response formats, check:
```python
resp = requests.post(f"{OLLAMA_URL}/api/embed", json={...})
print(resp.json())  # Inspect the actual response
```

### ChromaDB Vector Dimension Mismatch

Ensure the embedding model returns consistent vector dimensions. Check:
```python
vectors = integration.get_embeddings(["test"])
print(f"Vector dimension: {len(vectors[0])}")
```

### Large-Scale Ingestion

For processing many documents:
- Use batch processing (already implemented in `bulk_ingest.py`)
- Monitor memory usage
- Consider processing in smaller batches

## Example Workflows

### RAG (Retrieval-Augmented Generation)

```python
# 1. Query for relevant documents
results = integration.query("What is machine learning?", n_results=3)

# 2. Build context from results
context = "\n".join(results["documents"][0])

# 3. Generate with context
prompt = f"Based on the following context:\n{context}\n\nAnswer: What is machine learning?"
answer = integration.generate_text(prompt)
```

### Document Search

```python
# Search for similar documents
query = "vector databases"
results = integration.query(query, n_results=5)

for doc, metadata, distance in zip(
    results["documents"][0],
    results["metadatas"][0],
    results["distances"][0]
):
    print(f"Score: {1-distance:.2f} | Source: {metadata['source']}")
    print(f"Text: {doc[:200]}...\n")
```

## Additional Resources

- [Ollama Documentation](https://docs.ollama.com)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [LangChain Ollama Integration](https://docs.langchain.com/oss/python/integrations/text_embedding/ollama)

## License

MIT

# ollama
