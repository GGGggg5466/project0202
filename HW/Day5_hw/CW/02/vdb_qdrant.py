from typing import Any, Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue
)

class QdrantVDB:
    def __init__(self, host: str = "localhost", port: int = 6333, collection: str = "cw02", vector_size: int = 4096):
        self.client = QdrantClient(host=host, port=port)
        self.collection = collection
        self.vector_size = vector_size

    def recreate_collection(self) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
        )

    def upsert_points(self, points: List[Dict[str, Any]]) -> None:
        qpoints = [
            PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
            for p in points
        ]
        self.client.upsert(collection_name=self.collection, points=qpoints)

    def search(self, query_vector: List[float], top_k: int = 5, method: Optional[str] = None) -> List[Dict[str, Any]]:
        qfilter = None
        if method:
            qfilter = Filter(
                must=[FieldCondition(key="method", match=MatchValue(value=method))]
            )

        # ✅ qdrant-client 1.16.x 正規做法：query_points
        resp = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            query_filter=qfilter,
            with_payload=True,
            with_vectors=False,
        )

        results = []
        for p in resp.points:
            results.append({
                "id": p.id,
                "score": p.score,
                "payload": p.payload
            })
        return results
