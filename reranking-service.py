from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from typing import List, Optional
import uvicorn
import torch
from transformers import logging
from functools import lru_cache
import hashlib
from concurrent.futures import ThreadPoolExecutor
import os

# Set up logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

app = FastAPI()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_NAME = 'BAAI/bge-reranker-base'
#rerank_module = SentenceTransformer('mixedbread-ai/mxbai-rerank-xsmall-v1') # 50-70ms
#rerank_module = SentenceTransformer('jinaai/jina-reranker-v1-turbo-en') # 30-40ms
#rerank_module = SentenceTransformer('jinaai/jina-reranker-v2-base-multilingual') # 1s+
#rerank_module = SentenceTransformer('BAAI/bge-reranker-base') # 45-70 ms
#rerank_module = SentenceTransformer('BAAI/bge-reranker-v2-m3') #  100ms
#rerank_module = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') # 15-30ms
#sdadas/polish-reranker-large-mse

# Initialize the model
rerank_module = SentenceTransformer(MODEL_NAME).to(DEVICE)

# Set the number of worker threads
MAX_WORKERS = os.cpu_count() or 1
print("MAX_WORKERS: ", MAX_WORKERS)
# Create a ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

class RerankRequest(BaseModel):
    query: str = Field(..., description="The query to rerank against")
    texts: List[str] = Field(..., description="The list of texts to be reranked")
    truncate: bool = Field(False, description="Whether to truncate the text in the response")

class RerankResponse(BaseModel):
    index: int = Field(..., description="The original index of the text")
    score: float = Field(..., description="The similarity score")
    text: Optional[str] = Field(None, description="The original text (if not truncated)")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url}")
    response = await call_next(request)
    return response

@lru_cache(maxsize=1000)
def get_embedding(text: str):
    return rerank_module.encode([text])[0]

def hash_texts(texts: List[str]) -> str:
    return hashlib.md5(''.join(texts).encode()).hexdigest()

@app.post("/rerank", response_model=List[RerankResponse])
async def get_rerank_embeddings(request: RerankRequest):
    try:
        query = request.query
        texts = request.texts
        truncate = request.truncate

        # Use ThreadPoolExecutor for parallel processing
        future_query = executor.submit(get_embedding, query)
        future_texts = executor.submit(rerank_module.encode, texts)

        query_embedding = future_query.result()
        text_embeddings = future_texts.result()

        similarities = cos_sim(query_embedding, text_embeddings).flatten()

        reranked_results = [
            RerankResponse(
                index=i,
                score=float(score),
                text=text if not truncate else None
            )
            for i, (score, text) in enumerate(zip(similarities, texts))
        ]

        reranked_results.sort(key=lambda x: x.score, reverse=True)

        logger.info(f"Reranked {len(reranked_results)} results")
        return reranked_results

    except Exception as e:
        logger.error(f"Error in reranking: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
