from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List, Optional
import uvicorn
# from transformers.utils import logging

# logging.set_verbosity_info()
# logger = logging.get_logger("transformers")

app = FastAPI()

rerank_module = SentenceTransformer('mixedbread-ai/mxbai-rerank-xsmall-v1')

class RerankRequest(BaseModel):
    query: str
    texts: List[str]
    truncate: bool

class RerankResponse(BaseModel):
    index: int
    score: float
    text: Optional[str]

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Received request: {request.method} {request.url}")
    print(f"Request body: {await request.body()}")
    response = await call_next(request)
    return response

@app.post("/rerank", response_model=List[RerankResponse])
async def get_rerank_embeddings(request: RerankRequest):
    query = request.query
    texts = request.texts
    truncate = request.truncate

    # Perform reranking logic here
    embeddings = rerank_module.encode([query] + texts)
    query_embedding = embeddings[0]
    text_embeddings = embeddings[1:]

    similarities = cos_sim(query_embedding, text_embeddings).flatten()

    reranked_results = []
    for i, score in enumerate(similarities):
        reranked_result = RerankResponse(index=i, score=float(score), text=texts[i] if not truncate else None)
        reranked_results.append(reranked_result)

    reranked_results = sorted(reranked_results, key=lambda x: x.score, reverse=True)

    print(reranked_results)  
    return reranked_results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=True)
