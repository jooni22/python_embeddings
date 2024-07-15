from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
from transformers.utils import logging
from urllib.parse import urlparse, parse_qs
import torch
from concurrent.futures import ThreadPoolExecutor
import uvicorn
import asyncio

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = {
    "jinaai/jina-embeddings-v2-base-en": SentenceTransformer("jinaai/jina-embeddings-v2-base-en").to(device)
}

model_key_mapping = {
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jinaai/jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jina-base-en": "jinaai/jina-embeddings-v2-base-en"
}

executor = ThreadPoolExecutor(max_workers=8)
BATCH_SIZE = 32

class Item(BaseModel):
    input: Union[List[str], str]
    model: str

class Embedding(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[Embedding]
    model: str
    usage: dict = {"prompt_tokens": 0, "total_tokens": 0}

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    response = await call_next(request)
    return response

async def process_batch(texts, model):
    embeddings = model.encode(texts)
    return [Embedding(object="embedding", embedding=emb.tolist(), index=i) for i, emb in enumerate(embeddings)]

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: Request, item: Item):
    query_params = parse_qs(urlparse(str(request.url)).query)
    api_version = query_params.get("api-version")
    if api_version:
        logger.info(f"API Version: {api_version}")

    if isinstance(item.input, str):
        input_list = [item.input]
    else:
        input_list = item.input

    actual_model_key = model_key_mapping.get(item.model, item.model)
    if actual_model_key not in models:
        raise ValueError(f"Unsupported model: {item.model}. Supported models are: {', '.join(models.keys())}")

    selected_model = models[actual_model_key]

    batches = [input_list[i:i+BATCH_SIZE] for i in range(0, len(input_list), BATCH_SIZE)]
    embeddings = []
    for batch in batches:
        embeddings.extend(await process_batch(batch, selected_model))

    total_tokens = sum(len(text.split()) for text in input_list)

    response = EmbeddingResponse(
        object="list",
        data=embeddings,
        model=item.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    )

    logger.info(response.json())
    return response

# Check if we are in a Jupyter notebook by looking for a running event loop
if "get_ipython" in globals():
    import nest_asyncio
    from uvicorn import Config, Server

    nest_asyncio.apply()

    # Since uvicorn.run() starts an event loop, we need a non-blocking way to run the server in Jupyter
    config = Config(app=app, host="0.0.0.0", port=6000, log_level="info")
    server = Server(config)

    # Instead of uvicorn.run, use a new asyncio loop to start the server to avoid blocking
    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.serve())
else:
    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=6000, log_level="info")

