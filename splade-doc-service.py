from fastapi import FastAPI, Request
from pydantic import BaseModel, RootModel
from typing import List, Union
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import uvicorn

app = FastAPI()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the SPLADE model and tokenizer
model_name = 'naver/efficient-splade-VI-BT-large-doc'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
model.eval()  # Set the model to evaluation mode

class EmbedRequest(BaseModel):
    inputs: Union[str, List[str]]

class SparseValue(BaseModel):
    index: int
    value: float

class EmbedSparseResponse(RootModel):
    root: List[SparseValue]

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Received request: {request.method} {request.url}")
    print(f"Request body: {await request.body()}")
    response = await call_next(request)
    return response

@app.post("/embed_sparse", response_model=List[EmbedSparseResponse])
async def get_sparse_embedding(request: EmbedRequest):
    inputs = request.inputs

    # Ensure inputs is a list for consistent processing
    if isinstance(inputs, str):
        inputs = [inputs]

    # Tokenize the input and move to the appropriate device
    tokens = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    # Generate sparse embedding
    with torch.no_grad():
        output = model(**tokens)

    # Get the SPLADE sparse vector
    sparse_vec = torch.max(torch.log(1 + torch.relu(output.logits)) * tokens['attention_mask'].unsqueeze(-1), dim=1)[0]

    # Process the sparse vector on GPU
    non_zero_mask = sparse_vec[0] > 0  # Focus on the first (and only) item in the batch
    non_zero_indices = non_zero_mask.nonzero().squeeze(dim=1)
    non_zero_values = sparse_vec[0][non_zero_mask]

    # Sort on GPU
    sorted_indices = non_zero_values.argsort(descending=True)
    top_k = min(100, len(sorted_indices))  # Adjust the number as needed
    top_indices = non_zero_indices[sorted_indices[:top_k]]
    top_values = non_zero_values[sorted_indices[:top_k]]

    # Move only the necessary data to CPU
    top_indices = top_indices.cpu().tolist()
    top_values = top_values.cpu().tolist()

    # Create SparseValue objects
    sparse_values = [SparseValue(index=int(idx), value=float(val)) for idx, val in zip(top_indices, top_values)]

    response = EmbedSparseResponse(root=sparse_values)
    print(response.model_dump_json())
    return [response]  # Return the response in a double square bracket.

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4000, access_log=True)