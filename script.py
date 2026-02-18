import hashlib
import json
from typing import List, Dict, Any, Optional

import requests
from decouple import config
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class OpenRouterEmbedder:
    BASE_URL = "https://openrouter.ai/api/v1/embeddings"

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", timeout: int = 60):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def embed(self, text: str) -> list[float]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "input": text}

        r = requests.post(self.BASE_URL, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise Exception(f"OpenRouter error {r.status_code}: {r.text}")

        data = r.json()
        return data["data"][0]["embedding"]


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """
    Oddiy (char-based) chunker. Uzun matnlarni bo‘lib embedding qilish uchun.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def stable_int_id(*parts: str) -> int:
    """
    Matndan stabil 64-bit int id yasaydi (Qdrant id int bo‘la oladi).
    """
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16)


class QdrantService:
    def __init__(self, url: str, collection: str):
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self, vector_size: int) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            return

        # bor bo‘lsa vector size mosligini tekshir
        info = self.client.get_collection(self.collection)
        current_size = info.config.params.vectors.size
        if current_size != vector_size:
            raise ValueError(
                f"Collection '{self.collection}' vector size mismatch: "
                f"existing={current_size}, incoming={vector_size}"
            )

    def upsert_points(self, points: List[PointStruct]) -> None:
        self.client.upsert(collection_name=self.collection, points=points)


def insert_pages_json_to_qdrant(
    data: Dict[str, Any],
    *,
    embedder: OpenRouterEmbedder,
    qdrant: QdrantService,
    source: str = "unknown",
    max_chars: int = 1200,
    overlap: int = 200,
    batch_size: int = 64,
) -> int:
    pages = data.get("pages", [])
    total = 0
    buffer: List[PointStruct] = []
    collection_ready = False

    for page_idx, page in enumerate(pages):
        body = (page.get("body") or "").strip()
        if not body:
            continue

        chunks = chunk_text(body, max_chars=max_chars, overlap=overlap)

        for chunk_idx, chunk in enumerate(chunks):
            vec = embedder.embed(chunk)

            if not collection_ready:
                qdrant.ensure_collection(vector_size=len(vec))
                collection_ready = True

            pid = stable_int_id(source, str(page_idx), str(chunk_idx), chunk[:80])

            payload = {
                "source": source,
                "page_index": page_idx,
                "chunk_index": chunk_idx,
                "text": chunk,  # xohlasang saqla; xohlamasang olib tashla
            }

            buffer.append(PointStruct(id=pid, vector=vec, payload=payload))

            if len(buffer) >= batch_size:
                qdrant.upsert_points(buffer)
                total += len(buffer)
                buffer = []

    if buffer:
        qdrant.upsert_points(buffer)
        total += len(buffer)

    return total


if __name__ == "__main__":
    API_KEY = config("OPENROUTER_API_KEY")

    with open("test.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    embedder = OpenRouterEmbedder(api_key=API_KEY, model="text-embedding-3-small")
    qdrant = QdrantService(url="http://localhost:6333", collection="soffcrm")

    inserted = insert_pages_json_to_qdrant(
        data=data,
        embedder=embedder,
        qdrant=qdrant,
        source="soffcrm.uz",
        max_chars=1200,
        overlap=200,
        batch_size=32,
    )

    print("Inserted points:", inserted)
