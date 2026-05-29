import clip
import torch
import numpy as np
from PIL.Image import Image

class EmbeddingModel:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None

    def initialize(self):
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device, jit=False)

    def encode_image(self, img: Image):
        with torch.no_grad():
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            embeddings = self.model.encode_image(image_input).float()
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

        embeddings = embeddings.cpu().numpy().flatten()

        return embeddings
    
    def encode_text(self, query_text: str) -> np.ndarray:
        with torch.no_grad():
            text_input = clip.tokenize([query_text]).to(self.device)
            embeddings = self.model.encode_text(text_input).float()
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

        embeddings = embeddings.cpu().numpy().flatten()

        return embeddings
    
    def encode_batch_images(self, images: list[Image]):
        processed_images = []

        for img in images:
            processed_images.append(self.preprocess(img))

        image_input = torch.stack(processed_images).to(self.device)

        with torch.no_grad():
            batch_embeddings = self.model.encode_image(image_input).float()
            batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)

        embeddings_numpy = batch_embeddings.cpu().numpy()

        return embeddings_numpy