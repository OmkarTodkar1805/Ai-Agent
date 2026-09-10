"""Vector storage backed by an embedded Qdrant instance."""

import os
from dataclasses import dataclass
from typing import Optional

from qdrant_client import QdrantClient, models

from .chunking import Chunk
from .embeddings import VECTOR_SIZE, embed_documents, embed_query

DEFAULT_COLLECTION = "documents"
EMBED_BATCH_SIZE = int(os.environ.get("EMBED_BATCH_SIZE", "4"))


@dataclass
class Hit:
    """One retrieved chunk with its similarity score."""

    text: str
    page: int
    score: float
    source: str


class VectorStore:
    """Wraps Qdrant running in-process.

    Embedded mode keeps the deployment to a single container with no external
    service to provision, which is what makes the free tier viable. The same
    client talks to a hosted Qdrant by passing a url instead of a path, so
    moving to a managed cluster is a configuration change.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        url: Optional[str] = None,
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        self.collection = collection

        # A url reaches a Qdrant server, which is what a background worker needs:
        # a worker in its own process cannot see an index held in the web
        # process's memory. Path and memory modes keep single-container
        # deployments to one service.
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE, distance=models.Distance.COSINE
            ),
        )

    def reset(self) -> None:
        """Drop and recreate the collection, so a new upload replaces the old one."""
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)

        self._ensure_collection()

    def count(self) -> int:
        """Return how many chunks are stored."""
        return self.client.count(self.collection, exact=True).count

    def add(self, chunks: list[Chunk], source: str, batch_size: int = EMBED_BATCH_SIZE) -> int:
        """Embed and store chunks, returning how many were written.

        Batch size is the main lever on peak memory: the ONNX runtime grows its
        allocation arena with the batch, and this document costs 397MB of RSS at
        batch 4 against 705MB at batch 64. Small batches keep the app inside a
        512MB container for the price of a few more inference calls.
        """
        written = 0

        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = embed_documents([chunk.text for chunk in batch])

            self.client.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=chunk.index,
                        vector=vector,
                        payload={
                            "text": chunk.text,
                            "page": chunk.page,
                            "tokens": chunk.tokens,
                            "source": source,
                        },
                    )
                    for chunk, vector in zip(batch, vectors)
                ],
            )
            written += len(batch)

        return written

    def search(self, question: str, limit: int = 6) -> list[Hit]:
        """Return the closest chunks to a question, best first."""
        if self.count() == 0:
            return []

        results = self.client.query_points(
            collection_name=self.collection,
            query=embed_query(question),
            limit=limit,
            with_payload=True,
        ).points

        return [
            Hit(
                text=point.payload.get("text", ""),
                page=int(point.payload.get("page", 0)),
                score=float(point.score),
                source=str(point.payload.get("source", "")),
            )
            for point in results
            if point.payload
        ]
