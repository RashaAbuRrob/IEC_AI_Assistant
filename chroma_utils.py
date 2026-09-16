# -*- coding: utf-8 -*-
"""
ChromaDB setup: PersistentClient + a custom EmbeddingFunction that routes
through local Ollama bge-m3 (Chroma's built-in default embedder is NOT
bge-m3, so this override is required).
"""

from chromadb import Documents, EmbeddingFunction, Embeddings, PersistentClient

from ollama_utils import embed_text

COLLECTION_NAME = "iec_docs"


class BgeM3EF(EmbeddingFunction):
    """Calls Ollama's bge-m3 model once per input text. Chroma calls this
    both when documents are add()-ed and when a query_texts=[...] query
    is issued, so this single code path is what guarantees index-time and
    query-time embeddings come from the same model with the same call
    shape."""

    def __call__(self, input: Documents) -> Embeddings:
        out = []
        for t in input:
            out.append(embed_text(t))
        return out


def get_client(db_path: str) -> PersistentClient:
    return PersistentClient(path=db_path)


def get_collection(db_path: str):
    client = get_client(db_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=BgeM3EF(),
        metadata={"hnsw:space": "cosine"},
    )
    return collection
