from decouple import config
from fastapi import APIRouter
from app.api.schemas.test import (
    QuestionResponse,
    AnswerResponse
)
from embedder import OpenRouterEmbedder
from search import QdrantSemanticSearch

router = APIRouter()
API_KEY = config("OPENROUTER_API_KEY")
QDRANT_URL = config('QDRANT_URL')


@router.post("/question/", response_model=AnswerResponse)
async def test(body: QuestionResponse):

    embedder = OpenRouterEmbedder(api_key=API_KEY, model="text-embedding-3-small")
    searcher = QdrantSemanticSearch(
        qdrant_url=QDRANT_URL,
        collection="play_kb",
        embedder=embedder,
        text_key="text",
    )

    data = body.model_dump()
    res = searcher.ask(data['question'])
    return {
        "answer": res['text'],
        'status': "ok"
    }
