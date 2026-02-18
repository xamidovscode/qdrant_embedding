from __future__ import annotations

from typing import Any, Dict, Optional, List
import re
import requests
from qdrant_client import QdrantClient


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

        return r.json()["data"][0]["embedding"]


class QdrantSemanticSearch:
    """
    Qidiruv sifatini yaxshilangan variant:
      - limit=1 emas, top_k oladi
      - boilerplate (footer/menu/telefon) bo‘laklarni penalize/skip qiladi
      - score_threshold qo‘llaydi
      - qaytaradigan text payload ichidan text_key orqali olinadi
    """

    def __init__(
        self,
        qdrant_url: str,
        collection: str,
        embedder: OpenRouterEmbedder,
        text_key: str = "text",
    ):
        self.q = QdrantClient(url=qdrant_url)
        self.collection = collection
        self.embedder = embedder
        self.text_key = text_key

        # UI/footerdan keladigan shovqinlarni aniqlash uchun patternlar
        self._bad_substrings = [
            "Barcha huquqlar",
            "©",
            "SOFF CRM",
            "Demo olish",
            "Qo'ng'iroq",
            "Narxlarni ko'rish",
            "Bizning hamjamiyatimizga qo'shiling",
            "Kontakt",
            "Biz haqimizda",
            "Bosh sahifa",
        ]
        self._phone_re = re.compile(r"\+?\d[\d\s\-\(\)]{7,}\d")  # telefon
        self._only_short_tokens_re = re.compile(r"^(\W*\w{1,3}\W*){1,6}$")

    def _is_boilerplate(self, text: str) -> bool:
        if not text:
            return True

        t = text.strip()

        # juda qisqa / menu ko‘rinishida bo‘lsa
        if len(t) < 60:
            return True

        # telefon raqam ko‘p bo‘lsa
        if self._phone_re.search(t):
            return True

        # ko‘p shovqinli substringlar bo‘lsa
        low = t.lower()
        for s in self._bad_substrings:
            if s.lower() in low:
                return True

        # juda "menyu" ko‘rinishida (faqat qisqa tokenlar)
        if self._only_short_tokens_re.match(t.replace("\n", " ")):
            return True

        return False

    def _extract_text(self, payload: Dict[str, Any]) -> str:
        # afzal: text_key
        val = payload.get(self.text_key)
        if isinstance(val, str) and val.strip():
            return val.strip()

        # fallback lar
        for k in ("clean_text", "text", "body", "answer", "question"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        return ""

    def ask(
            self,
            question: str,
            *,
            top_k: int = 10,
            score_threshold: Optional[float] = None,  # default: o‘chirilgan
    ) -> Dict[str, Any]:
        vector = self.embedder.embed(question)

        res = self.q.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        if not res.points:
            return {"found": False, "text": None, "score": None, "payload": None}

        best = None
        best_score = -1e9

        for p in res.points:
            score = float(p.score) if p.score is not None else 0.0
            if score_threshold is not None and score < score_threshold:
                continue

            payload = p.payload or {}
            text = self._extract_text(payload)
            if not text:
                continue

            # penalti: boilerplate bo‘lsa biroz kamaytiramiz, lekin tashlab yubormaymiz
            penalized = score - (0.10 if self._is_boilerplate(text) else 0.0)

            if penalized > best_score:
                best_score = penalized
                best = (text, score, payload)

        # agar filtrlar sababli hech narsa tanlanmasa, baribir top1 qaytaramiz
        if best is None:
            p0 = res.points[0]
            payload0 = p0.payload or {}
            text0 = self._extract_text(payload0)
            score0 = float(p0.score) if p0.score is not None else None
            return {"found": True, "text": text0, "score": score0, "payload": payload0}

        text, score, payload = best
        return {"found": True, "text": text, "score": score, "payload": payload}
