from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantService:
    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "play_kb",
        vector_size: int = 4,
    ):
        self.url = url
        self.collection = collection
        self.vector_size = vector_size
        self.client = QdrantClient(url=self.url)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    def upsert(self, items: List[Dict[str, Any]]) -> None:
        for item in items:
            if len(item["vector"]) != self.vector_size:
                raise ValueError(
                    f"Vector size mismatch id={item['id']}"
                )

        points = [
            PointStruct(
                id=item["id"],
                vector=item["vector"],
                payload=item.get("payload", {}),
            )
            for item in items
        ]

        self.client.upsert(
            collection_name=self.collection,
            points=points,
        )

    def list_points(self, limit: int = 20) -> None:
        points, _ = self.client.scroll(
            collection_name=self.collection,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        for p in points:
            print(f"id={p.id} payload={p.payload}")


qdrant = QdrantService()
qdrant.ensure_collection()

data = [
    {
        "id": 1,
        "vector": [1.0, 0.0, 0.0, 0.0],
        "payload": {"question": "kpiss nima?", "answer": "KPI — samaradorlik"},
    },
    {
        "id": 2,
        "vector": [0.9, 0.1, 0.0, 0.0],
        "payload": {"question": "kpi degani nima", "answer": "KPI — performance"},
    },
]

qdrant.upsert(data)

