import torch, clip, cv2
import numpy as np
from PIL import Image
from guard.core.interfaces import VectorizerInterface
from guard.core.entities import VideoFrame, VectorEmbedding


class CLIPVectorizer(VectorizerInterface):
    def __init__(self):
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/16", device=self.device, jit=False)

    def encode_image(self, frame: VideoFrame) -> VectorEmbedding:
        img = Image.fromarray(cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            embeddings = self.model.encode_image(image_input).float()
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

        embeddings = embeddings.cpu().numpy().flatten()

        return VectorEmbedding(embeddings=embeddings, metadata={})
    
    def encode_text(self, query_text: str) -> np.ndarray:
        with torch.no_grad():
            text_input = clip.tokenize([query_text]).to(self.device)
            embeddings = self.model.encode_text(text_input).float()
            embeddings /= embeddings.norm(dim=-1, keepdim=True)

        embeddings = embeddings.cpu().numpy().flatten()

        return embeddings