#!/usr/bin/env python3
"""
Ollama + ChromaDB Integration Example

This script demonstrates how to:
1. Generate embeddings from Ollama
2. Store them in ChromaDB
3. Query the database for similar documents
"""

import requests
from typing import List, Optional
import json

# Monkey patch for pydantic compatibility with chromadb
import os
# Set default values for chromadb settings to avoid validation errors
os.environ.setdefault('CLICKHOUSE_HOST', '')
os.environ.setdefault('CLICKHOUSE_PORT', '')
os.environ.setdefault('CHROMA_SERVER_HOST', '')
os.environ.setdefault('CHROMA_SERVER_HTTP_PORT', '')
os.environ.setdefault('CHROMA_SERVER_GRPC_PORT', '')

import pydantic
import pydantic_settings

# Make BaseSettings available in pydantic namespace for chromadb compatibility
# Directly set it to avoid triggering __getattr__
pydantic.BaseSettings = pydantic_settings.BaseSettings

import chromadb


OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "embeddinggemma"  # or another embed model you have pulled


class OllamaChromaIntegration:
    """Integration class for Ollama embeddings and ChromaDB storage."""
    
    def __init__(
        self,
        ollama_url: str = OLLAMA_URL,
        embed_model: str = EMBED_MODEL,
        collection_name: str = "my_docs",
        persist_directory: str = "./chroma_db"
    ):
        self.ollama_url = ollama_url
        self.embed_model = embed_model
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        # Initialize ChromaDB client
        try:
            # Try new API first (ChromaDB 0.4+)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        except (AttributeError, TypeError):
            # Fallback to old API - use Client directly without Settings
            try:
                self.client = chromadb.Client()
            except Exception:
                # Last resort: try with minimal settings
                import os
                os.environ.setdefault('CHROMA_DB_IMPL', 'chromadb.db')
                self.client = chromadb.Client()
        
        # Get or create collection
        try:
            # Try to get existing collection first
            self.collection = self.client.get_collection(collection_name)
            print(f"Using existing collection: {collection_name}")
        except Exception:
            # Create new collection - use get_or_create_collection for better compatibility
            try:
                self.collection = self.client.get_or_create_collection(collection_name)
            except AttributeError:
                # Fallback to create_collection
                self.collection = self.client.create_collection(collection_name)
            print(f"Created new collection: {collection_name}")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings from Ollama for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        resp = requests.post(
            f"{self.ollama_url}/api/embed",
            json={
                "model": self.embed_model,
                "input": texts
            }
        )
        resp.raise_for_status()
        data = resp.json()
        
        # Ollama returns embeddings in `embeddings` (list of vectors)
        vectors = data.get("embeddings") or data.get("data")
        if vectors is None:
            raise RuntimeError(f"Unexpected Ollama /api/embed response: {data}")
        
        return vectors
    
    def add_documents(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None
    ):
        """
        Add documents to ChromaDB with embeddings from Ollama.
        
        Args:
            texts: List of document texts
            ids: Optional list of document IDs (auto-generated if None)
            metadatas: Optional list of metadata dicts
        """
        if ids is None:
            ids = [f"doc-{i}" for i in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{"source": "ollama_demo"} for _ in texts]
        
        # Get embeddings from Ollama
        print(f"Generating embeddings for {len(texts)} documents...")
        vectors = self.get_embeddings(texts)
        
        # Add to ChromaDB - try collection.add first, fallback to client API
        try:
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=vectors,
                metadatas=metadatas
            )
        except AttributeError:
            # Fallback: use client API directly
            self.client.get_collection(self.collection_name).add(
                ids=ids,
                documents=texts,
                embeddings=vectors,
                metadatas=metadatas
            )
        print(f"Added {len(texts)} documents to ChromaDB")
    
    def query(
        self,
        query_text: str,
        n_results: int = 3,
        return_documents: bool = True
    ) -> dict:
        """
        Query ChromaDB for similar documents.
        
        Args:
            query_text: Query text to search for
            n_results: Number of results to return
            return_documents: Whether to return document texts
            
        Returns:
            Dictionary with query results
        """
        # Get embedding for query
        print(f"Generating embedding for query: '{query_text}'")
        qvec = self.get_embeddings([query_text])[0]
        
        # Query ChromaDB
        results = self.collection.query(
            n_results=n_results,
            query_embeddings=[qvec],
            include=["documents", "metadatas", "distances"] if return_documents else ["metadatas", "distances"]
        )
        
        return results
    
    def generate_text(self, prompt: str, model: str = "gemma3:12b") -> str:
        """
        Generate text using Ollama (for completion tasks).
        
        Args:
            prompt: Text prompt
            model: Model to use for generation
            
        Returns:
            Generated text
        """
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            }
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


def main():
    """Example usage of OllamaChromaIntegration."""
    
    # Initialize integration
    integration = OllamaChromaIntegration()
    
    # Example documents
    texts = [
        "Llamas are members of the camelid family.",
        "ChromaDB is a lightweight vector DB for embeddings and retrieval.",
        "Ollama provides local LLM inference and embedding generation.",
        "Vector databases enable semantic search and RAG applications."
    ]
    
    # Add documents to ChromaDB
    print("\n=== Adding Documents ===")
    integration.add_documents(texts)
    
    # Query the database
    print("\n=== Querying Database ===")
    query_text = "What is a llama?"
    results = integration.query(query_text, n_results=3)
    
    print(f"\nQuery: '{query_text}'")
    print(f"\nResults:")
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ), 1):
        print(f"\n{i}. Distance: {distance:.4f}")
        print(f"   Document: {doc}")
        print(f"   Metadata: {metadata}")
    
    # Example: Generate text with the generation model
    print("\n=== Text Generation Example ===")
    prompt = "Summarize the following: Ollama + Chroma integration steps."
    generated = integration.generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")


if __name__ == "__main__":
    main()

