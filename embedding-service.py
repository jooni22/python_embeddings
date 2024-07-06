from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, models
from typing import List, Union
import uvicorn
from transformers.utils import logging


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

logging.set_verbosity_info()
logger = logging.get_logger("transformers")

app = FastAPI()

dimensions = 512
# Initialize both models
models = {
    "baai/bge-m3": SentenceTransformer('baai/bge-m3'),
    "mixedbread-ai/mxbai-embed-large-v1": SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
}

class Item(BaseModel):
    input: Union[List[str], str]
    model: str

class Embedding(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]

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
async def get_embeddings(item: Item):
    if isinstance(item.input, str):
        input_list = [item.input]
    else:
        input_list = item.input

    if item.model not in models:
        raise ValueError(f"Unsupported model: {item.model}. Supported models are: {', '.join(models.keys())}")

    embeddings = []
    total_tokens = 0
    selected_model = models[item.model]
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
    uvicorn.run(app, host="0.0.0.0", port=9200, access_log=True)
