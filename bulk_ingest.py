#!/usr/bin/env python3
"""
Bulk Document Ingestion Script

Processes documents from a folder and ingests them into ChromaDB using Ollama embeddings.
Supports .txt, .md, .pdf files (requires PyPDF2 for PDF).
"""

import os
import sys
from pathlib import Path
from typing import List, Dict
from ollama_chroma import OllamaChromaIntegration
import argparse


def read_text_file(file_path: Path) -> str:
    """Read a text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_markdown_file(file_path: Path) -> str:
    """Read a markdown file."""
    return read_text_file(file_path)  # Same as text for now


def read_pdf_file(file_path: Path) -> str:
    """Read a PDF file (requires PyPDF2)."""
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF support. Install with: pip install PyPDF2")
    
    text = ""
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text


def process_file(file_path: Path) -> str:
    """Process a file and return its text content."""
    suffix = file_path.suffix.lower()
    
    if suffix == '.txt':
        return read_text_file(file_path)
    elif suffix == '.md':
        return read_markdown_file(file_path)
    elif suffix == '.pdf':
        return read_pdf_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk in characters
        overlap: Overlap between chunks in characters
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks


def ingest_folder(
    folder_path: str,
    collection_name: str = "bulk_docs",
    chunk_size: int = 1000,
    overlap: int = 200,
    file_extensions: List[str] = None,
    batch_size: int = 10
):
    """
    Ingest all documents from a folder into ChromaDB.
    
    Args:
        folder_path: Path to folder containing documents
        collection_name: Name of ChromaDB collection
        chunk_size: Size of text chunks
        overlap: Overlap between chunks
        file_extensions: List of file extensions to process (default: ['.txt', '.md', '.pdf'])
        batch_size: Number of documents to process in each batch
    """
    if file_extensions is None:
        file_extensions = ['.txt', '.md', '.pdf']
    
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Initialize integration
    integration = OllamaChromaIntegration(collection_name=collection_name)
    
    # Find all files
    files = []
    for ext in file_extensions:
        files.extend(folder.rglob(f"*{ext}"))
    
    if not files:
        print(f"No files found with extensions {file_extensions} in {folder_path}")
        return
    
    print(f"Found {len(files)} files to process")
    
    # Process files in batches
    all_texts = []
    all_ids = []
    all_metadatas = []
    doc_id = 0
    
    for file_path in files:
        try:
            print(f"Processing: {file_path}")
            text = process_file(file_path)
            
            # Chunk the text
            chunks = chunk_text(text, chunk_size, overlap)
            
            # Add chunks with metadata
            for i, chunk in enumerate(chunks):
                all_texts.append(chunk)
                all_ids.append(f"doc-{doc_id}-chunk-{i}")
                all_metadatas.append({
                    "source": str(file_path),
                    "file_name": file_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                })
            
            doc_id += 1
            print(f"  -> Created {len(chunks)} chunks")
            
            # Process in batches to avoid memory issues
            if len(all_texts) >= batch_size:
                print(f"\nProcessing batch of {len(all_texts)} chunks...")
                integration.add_documents(
                    texts=all_texts,
                    ids=all_ids,
                    metadatas=all_metadatas
                )
                all_texts = []
                all_ids = []
                all_metadatas = []
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)
            continue
    
    # Process remaining documents
    if all_texts:
        print(f"\nProcessing final batch of {len(all_texts)} chunks...")
        integration.add_documents(
            texts=all_texts,
            ids=all_ids,
            metadatas=all_metadatas
        )
    
    print(f"\n✅ Ingestion complete! Processed {doc_id} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingest documents from a folder into ChromaDB using Ollama embeddings"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Path to folder containing documents"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="bulk_docs",
        help="ChromaDB collection name (default: bulk_docs)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size in characters (default: 1000)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".txt", ".md", ".pdf"],
        help="File extensions to process (default: .txt .md .pdf)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of documents to process per batch (default: 10)"
    )
    
    args = parser.parse_args()
    
    ingest_folder(
        folder_path=args.folder,
        collection_name=args.collection,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        file_extensions=args.extensions,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()

