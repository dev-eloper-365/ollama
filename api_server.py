#!/usr/bin/env python3
"""
Simple API server for Ollama + ChromaDB integration
Provides REST endpoints for embeddings and document management
"""

from flask import Flask, request, jsonify
import requests
import os

# Monkey patch for pydantic compatibility with chromadb
os.environ.setdefault('CLICKHOUSE_HOST', '')
os.environ.setdefault('CLICKHOUSE_PORT', '')
os.environ.setdefault('CHROMA_SERVER_HOST', '')
os.environ.setdefault('CHROMA_SERVER_HTTP_PORT', '')
os.environ.setdefault('CHROMA_SERVER_GRPC_PORT', '')

import pydantic
import pydantic_settings
pydantic.BaseSettings = pydantic_settings.BaseSettings

import chromadb

app = Flask(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "embeddinggemma")
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Initialize ChromaDB client
client = None
try:
    # Try new API first
    if hasattr(chromadb, 'PersistentClient'):
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        print(f"✅ ChromaDB initialized at {PERSIST_DIR}")
    else:
        # Fallback to old API
        try:
            from chromadb.config import Settings
            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb",
                persist_directory=PERSIST_DIR
            ))
            print(f"✅ ChromaDB initialized (legacy API) at {PERSIST_DIR}")
        except:
            client = None
except Exception as e:
    print(f"⚠️  ChromaDB initialization issue: {e}")
    print("   Using memory store as fallback")
    client = None

# Store collections in memory as fallback
memory_store = {}


def get_embeddings(texts):
    """Get embeddings from Ollama"""
    if isinstance(texts, str):
        texts = [texts]
    
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": texts
        },
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    vectors = data.get("embeddings") or data.get("data")
    if vectors is None:
        raise RuntimeError(f"Unexpected Ollama response: {data}")
    return vectors


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Test Ollama connection
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_status = "connected" if resp.status_code == 200 else "disconnected"
    except:
        ollama_status = "disconnected"
    
    return jsonify({
        "status": "ok",
        "ollama": ollama_status,
        "chromadb": "available" if client else "unavailable",
        "embed_model": EMBED_MODEL
    })


@app.route('/api/embed', methods=['POST'])
def embed():
    """Generate embeddings for text(s)"""
    try:
        data = request.json
        texts = data.get('texts') or data.get('input', [])
        
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
        vectors = get_embeddings(texts)
        
        return jsonify({
            "embeddings": vectors,
            "count": len(vectors),
            "dimension": len(vectors[0]) if vectors else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/collections', methods=['GET'])
def list_collections():
    """List all collections"""
    if not client:
        return jsonify({
            "collections": list(memory_store.keys()),
            "note": "Using memory store (ChromaDB unavailable)"
        })
    
    try:
        collections = client.list_collections()
        return jsonify({
            "collections": [c.name for c in collections]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/collections/<collection_name>', methods=['POST'])
def create_collection(collection_name):
    """Create a new collection"""
    if not client:
        if collection_name not in memory_store:
            memory_store[collection_name] = []
        return jsonify({
            "status": "created",
            "collection": collection_name,
            "note": "Using memory store"
        })
    
    try:
        collection = client.get_or_create_collection(collection_name)
        return jsonify({
            "status": "created",
            "collection": collection_name,
            "count": collection.count()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/collections/<collection_name>/add', methods=['POST'])
def add_documents(collection_name):
    """Add documents to a collection"""
    try:
        data = request.json
        texts = data.get('texts') or data.get('documents', [])
        ids = data.get('ids')
        metadatas = data.get('metadatas')
        
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
        
        # Generate embeddings
        vectors = get_embeddings(texts)
        
        # Auto-generate IDs if not provided
        if not ids:
            ids = [f"doc-{i}" for i in range(len(texts))]
        
        if not metadatas:
            metadatas = [{}] * len(texts)
        
        # Try ChromaDB first
        if client:
            try:
                collection = client.get_or_create_collection(collection_name)
                # Use upsert which is more reliable
                collection.upsert(
                    ids=ids,
                    documents=texts,
                    embeddings=vectors,
                    metadatas=metadatas
                )
                return jsonify({
                    "status": "added",
                    "collection": collection_name,
                    "count": len(texts),
                    "storage": "chromadb"
                })
            except Exception as e:
                print(f"ChromaDB error: {e}, falling back to memory")
        
        # Fallback to memory store
        if collection_name not in memory_store:
            memory_store[collection_name] = []
        
        for i, (text, vec, meta, doc_id) in enumerate(zip(texts, vectors, metadatas, ids)):
            memory_store[collection_name].append({
                "id": doc_id,
                "text": text,
                "embedding": vec,
                "metadata": meta
            })
        
        return jsonify({
            "status": "added",
            "collection": collection_name,
            "count": len(texts),
            "storage": "memory"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/collections/<collection_name>/query', methods=['POST'])
def query_collection(collection_name):
    """Query a collection"""
    try:
        data = request.json
        query_text = data.get('query') or data.get('text')
        n_results = data.get('n_results', 3)
        
        if not query_text:
            return jsonify({"error": "No query provided"}), 400
        
        # Get query embedding
        query_vector = get_embeddings([query_text])[0]
        
        # Try ChromaDB first
        if client:
            try:
                collection = client.get_collection(collection_name)
                results = collection.query(
                    query_embeddings=[query_vector],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                return jsonify({
                    "results": [
                        {
                            "document": doc,
                            "metadata": meta,
                            "distance": dist
                        }
                        for doc, meta, dist in zip(
                            results["documents"][0],
                            results["metadatas"][0],
                            results["distances"][0]
                        )
                    ],
                    "storage": "chromadb"
                })
            except Exception as e:
                print(f"ChromaDB query error: {e}, falling back to memory")
        
        # Fallback: simple cosine similarity in memory
        if collection_name not in memory_store:
            return jsonify({"error": "Collection not found"}), 404
        
        import numpy as np
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        docs = memory_store[collection_name]
        similarities = [
            (doc, cosine_similarity(query_vector, doc["embedding"]))
            for doc in docs
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = [
            {
                "document": doc["text"],
                "metadata": doc["metadata"],
                "distance": 1 - sim  # Convert similarity to distance
            }
            for doc, sim in similarities[:n_results]
        ]
        
        return jsonify({
            "results": results,
            "storage": "memory"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/generate', methods=['POST'])
def generate():
    """Generate text using Ollama"""
    try:
        data = request.json
        prompt = data.get('prompt')
        model = data.get('model', 'gemma3:12b')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400
        
        resp = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        resp.raise_for_status()
        result = resp.json()
        
        return jsonify({
            "response": result.get("response", ""),
            "model": model
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    print(f"\n🚀 Starting Ollama + ChromaDB API server on http://localhost:{port}")
    print(f"📝 Embed model: {EMBED_MODEL}")
    print(f"🔗 Ollama URL: {OLLAMA_URL}")
    print(f"\nAvailable endpoints:")
    print(f"  GET  /health - Health check")
    print(f"  POST /api/embed - Generate embeddings")
    print(f"  GET  /api/collections - List collections")
    print(f"  POST /api/collections/<name> - Create collection")
    print(f"  POST /api/collections/<name>/add - Add documents")
    print(f"  POST /api/collections/<name>/query - Query collection")
    print(f"  POST /api/generate - Generate text\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)

