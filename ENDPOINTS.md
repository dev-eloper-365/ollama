# API Endpoints

## Base URL
```
http://localhost:8080
```

## Endpoints

### 1. Health Check
**GET** `/health`

Check server and Ollama status.

**Response:**
```json
{
  "status": "ok",
  "ollama": "connected",
  "chromadb": "available",
  "embed_model": "embeddinggemma"
}
```

**Example:**
```bash
curl http://localhost:8080/health
```

---

### 2. Generate Embeddings
**POST** `/api/embed`

Generate embeddings for one or more texts using Ollama.

**Request Body:**
```json
{
  "texts": ["Hello world", "Another text"]
}
```
or
```json
{
  "input": ["Hello world"]
}
```

**Response:**
```json
{
  "embeddings": [[...], [...]],
  "count": 2,
  "dimension": 768
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world", "Test embedding"]}'
```

---

### 3. List Collections
**GET** `/api/collections`

List all available collections.

**Response:**
```json
{
  "collections": ["test", "my_docs"]
}
```

**Example:**
```bash
curl http://localhost:8080/api/collections
```

---

### 4. Create Collection
**POST** `/api/collections/<collection_name>`

Create a new collection.

**Response:**
```json
{
  "status": "created",
  "collection": "test",
  "count": 0
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/collections/my_docs \
  -H "Content-Type: application/json" \
  -d '{}'
```

---

### 5. Add Documents
**POST** `/api/collections/<collection_name>/add`

Add documents to a collection with embeddings.

**Request Body:**
```json
{
  "texts": ["Document 1", "Document 2"],
  "ids": ["doc-1", "doc-2"],  // optional
  "metadatas": [{"source": "web"}, {"source": "book"}]  // optional
}
```

**Response:**
```json
{
  "status": "added",
  "collection": "test",
  "count": 2,
  "storage": "memory"  // or "chromadb"
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/collections/test/add \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Llamas are members of the camelid family",
      "ChromaDB is a vector database"
    ]
  }'
```

---

### 6. Query Collection
**POST** `/api/collections/<collection_name>/query`

Query a collection for similar documents.

**Request Body:**
```json
{
  "query": "What is a llama?",
  "n_results": 3
}
```

**Response:**
```json
{
  "results": [
    {
      "document": "Llamas are members of the camelid family",
      "metadata": {},
      "distance": 0.432
    }
  ],
  "storage": "memory"
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/collections/test/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is a llama?",
    "n_results": 2
  }'
```

---

### 7. Generate Text
**POST** `/api/generate`

Generate text using Ollama's language model.

**Request Body:**
```json
{
  "prompt": "Summarize the following: Ollama + Chroma integration steps.",
  "model": "gemma3:12b"  // optional, defaults to gemma3:12b
}
```

**Response:**
```json
{
  "response": "Generated text here...",
  "model": "gemma3:12b"
}
```

**Example:**
```bash
curl -X POST http://localhost:8080/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is machine learning?"
  }'
```

---

## Quick Test Script

```bash
# 1. Health check
curl http://localhost:8080/health

# 2. Generate embeddings
curl -X POST http://localhost:8080/api/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello world"]}'

# 3. Create collection and add documents
curl -X POST http://localhost:8080/api/collections/test/add \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Llamas are members of the camelid family"]}'

# 4. Query collection
curl -X POST http://localhost:8080/api/collections/test/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is a llama?", "n_results": 1}'
```

---

## Server Status

The API server is running on **http://localhost:8080**

To start the server:
```bash
cd /Users/patel/Code/Ollama
source venv/bin/activate
python api_server.py
```

To run in background:
```bash
PORT=8080 python api_server.py > /tmp/ollama_api.log 2>&1 &
```

