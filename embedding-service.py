from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, models
from typing import List, Union
import uvicorn
from transformers.utils import logging
from urllib.parse import urlparse, parse_qs

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

app = FastAPI()

dimensions = 512
# Initialize models including the new one
models = {
    "baai/bge-m3": SentenceTransformer('baai/bge-m3'),
#    "mixedbread-ai/mxbai-embed-large-v1": SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1"),
    "jinaai/jina-embeddings-v2-base-en": SentenceTransformer("jinaai/jina-embeddings-v2-base-en")
}

# Mapping for model keys
model_key_mapping = {
    "bge-m3": "baai/bge-m3",
    "baai/bge-m3": "baai/bge-m3",
    # "mixedbread-ai/mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
    # "mxbai-embed-large-v1": "mixedbread-ai/mxbai-embed-large-v1",
    "jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jinaai/jina-embeddings-v2-base-en": "jinaai/jina-embeddings-v2-base-en",
    "jina-base-en": "jinaai/jina-embeddings-v2-base-en"    
}

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
    print(f"Received request: {request.method} {request.url}")
    print(f"Request body: {await request.body()}")
    response = await call_next(request)
    return response

@app.post("/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: Request, item: Item):
    query_params = parse_qs(urlparse(str(request.url)).query)
    api_version = query_params.get("api-version")
    if api_version:
        print(f"API version: {api_version}")
        
    if isinstance(item.input, str):
        input_list = [item.input]
    else:
        input_list = item.input

    # Map model key if necessary
    actual_model_key = model_key_mapping.get(item.model, item.model)

    if actual_model_key not in models:
        raise ValueError(f"Unsupported model: {item.model}. Supported models are: {', '.join(models.keys())}")

    embeddings = []
    total_tokens = 0
    selected_model = models[actual_model_key]
    for i, text in enumerate(input_list):
        embedding = selected_model.encode(text)
        embeddings.append(Embedding(object="embedding", embedding=embedding.tolist(), index=i))
        total_tokens += len(text.split())

    response = EmbeddingResponse(
        object="list",
        data=embeddings,
        model=item.model,
        usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
    )
    print(response.json()) 
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6000, access_log=True)