import io
import os
import base64

import torch
import clip
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

MODEL_NAME = os.environ.get("CLIP_MODEL", "ViT-B/32")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load(MODEL_NAME, device=DEVICE, jit=False)

app = FastAPI(title="CLIP Embedding Service")

class TextRequest(BaseModel):
    text: str

class ImagesRequest(BaseModel):
    images: list[str]

def _decode_image(b64_str: str) -> Image.Image:
    try:
        image_bytes = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image payload")

@app.post("/encode/text")
def encode_text(request: TextRequest):
    with torch.no_grad():
        text_input = clip.tokenize([request.text]).to(DEVICE)
        embedding = model.encode_text(text_input).float()
        embedding /= embedding.norm(dim=-1, keepdim=True)

    return {"embedding": embedding.cpu().numpy().flatten().tolist()}

@app.post("/encode/images")
def encode_images(request: ImagesRequest):
    if not request.images:
        return {"embeddings": []}

    images = [preprocess(_decode_image(b64)) for b64 in request.images]
    image_input = torch.stack(images).to(DEVICE)

    with torch.no_grad():
        embeddings = model.encode_image(image_input).float()
        embeddings /= embeddings.norm(dim=-1, keepdim=True)

    return {"embeddings": embeddings.cpu().numpy().tolist()}

@app.get("/healthcheck")
def healthcheck():
    return {"status": "OK"}
