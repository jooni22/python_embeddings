from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from typing import List, Union, Dict
from transformers.utils import logging
from urllib.parse import urlparse, parse_qs
import torch
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import asyncio

# Set up logging
logging.set_verbosity_info()
logger = logging.get_logger("transformers")

app = FastAPI()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

# Model configurations
MODEL_KEY_MAPPING = {
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jinaai/jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jina-base-en": "jinaai/jina-embeddings-v2-base-en"
}

# Initialize models lazily
MODELS: Dict[str, SentenceTransformer] = {}

def get_model(model_name: str) -> SentenceTransformer:
    actual_model_key = MODEL_KEY_MAPPING.get(model_name, model_name)
    if actual_model_key not in MODELS:
        MODELS[actual_model_key] = SentenceTransformer(actual_model_key).to(DEVICE)
    return MODELS[actual_model_key]

class EmbeddingRequest(BaseModel):
    input: Union[List[str], str] = Field(..., title="Text to be embedded")
    model: str = Field(..., title="Model name")

class Embedding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "total_tokens": 0})

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response

def process_batch(texts: List[str], model: SentenceTransformer) -> List[Embedding]:
    embeddings = model.encode(texts)
    return [Embedding(object="embedding", embedding=emb.tolist(), index=i) for i, emb in enumerate(embeddings)]

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: Request, item: EmbeddingRequest):
    query_params = parse_qs(urlparse(str(request.url)).query)
    api_version = query_params.get("api-version")
    if api_version:
        logger.info(f"API Version: {api_version[0]}")

    try:
        model = get_model(item.model)
    except KeyError:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {item.model}")

    input_list = [item.input] if isinstance(item.input, str) else item.input

    batches = [input_list[i:i+BATCH_SIZE] for i in range(0, len(input_list), BATCH_SIZE)]
    embeddings = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_batch, batch, model) for batch in batches]
        for future in futures:
            embeddings.extend(future.result())

    total_tokens = sum(len(text.split()) for text in input_list)

    response = EmbeddingResponse(
        data=embeddings,
        model=item.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    )

    logger.info(f"Response: {response.json()}")
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000, log_level="info")
