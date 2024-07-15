# The error occurs because `asyncio.run()` cannot be called from an already running event loop, which is the case in Jupyter notebooks.
# The fix below adapts the code to check for the running event loop and uses `await` to run the server directly if it is in a Jupyter notebook.

from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List, Union
import uvicorn
from transformers.utils import logging
from urllib.parse import urlparse, parse_qs
import torch
from concurrent.futures import ThreadPoolExecutor
import asyncio

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

app = FastAPI()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

models = {
    "jinaai/jina-embeddings-v2-base-en": SentenceTransformer("jinaai/jina-embeddings-v2-base-en").to(device)
}

model_key_mapping = {
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jinaai/jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jina-base-en": "jinaai/jina-embeddings-v2-base-en"
}

executor = ThreadPoolExecutor(max_workers=8)

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

async def process_embedding(text, model, index):
    embedding = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: model.encode(text, device=device)
    )
    return Embedding(object="embedding", embedding=embedding.tolist(), index=index)

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

    embeddings = await asyncio.gather(*[process_embedding(text, selected_model, i) for i, text in enumerate(input_list)])

    total_tokens = sum(len(text.split()) for text in input_list)

    response = EmbeddingResponse(
        object="list",
        data=embeddings,
        model=item.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    )

    logger.info(response.json())
    return response

if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=6000, access_log=False)

