from __future__ import annotations

from typing import List, Union

from sentence_transformers import SentenceTransformer


class E5Embedder:
    """
    multilingual-e5-small uchun oddiy embedder.

    E5 oilasida yaxshi amaliyot:
      - query uchun:  "query: <matn>"
      - hujjat/passage uchun: "passage: <matn>"

    normalize_embeddings=True -> cosine similarity uchun juda qulay (vektorlar normallashadi).
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        prefixed = [f"passage: {t}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        vec = self.model.encode(
            f"query: {text}",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vec.tolist()


def main():
    emb = E5Embedder()

    passages = [
        "KPI — samaradorlik ko‘rsatkichlari (Key Performance Indicators).",
        "Pricing — mahsulot yoki xizmatga narx belgilash jarayoni.",
        "Django DurationField — vaqt davomiyligini saqlash uchun field.",
    ]

    query = "kpi nima degani?"

    p_vecs = emb.embed_passages(passages)
    q_vec = emb.embed_query(query)

    print("Model:", emb.model_name)
    print("Passage vector dim:", len(p_vecs[0]))
    print("Query vector dim:", len(q_vec))

    print("\nQuery vector (first 12 numbers):")
    print([round(x, 6) for x in q_vec[:12]])

    print("\nPassage[0] vector (first 12 numbers):")
    print([round(x, 6) for x in p_vecs[0][:12]])


if __name__ == "__main__":
    main()
