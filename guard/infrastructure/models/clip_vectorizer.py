import torch, clip, cv2, os
import numpy as np
from PIL import Image
from guard.core.interfaces import VectorizerInterface
from guard.core.entities import VideoFrame, VectorEmbedding

class CLIPVectorizer(VectorizerInterface):
    def __init__(self):
        super().__init__()

        self.modelName = os.environ.get("CLIP_MODEL", "ViT-B/32")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(self.modelName, device=self.device, jit=False)

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
    
    def encode_batch_images(self, frames: list[VideoFrame]) -> list[VectorEmbedding]:
        if not frames:
            return []

        processed_images = []

        for frame in frames:
            img = Image.fromarray(cv2.cvtColor(frame.data, cv2.COLOR_BGR2RGB))
            processed_images.append(self.preprocess(img))

        image_input = torch.stack(processed_images).to(self.device)

        with torch.no_grad():
            batch_embeddings = self.model.encode_image(image_input).float()
            batch_embeddings /= batch_embeddings.norm(dim=-1, keepdim=True)

        embeddings_numpy = batch_embeddings.cpu().numpy()

        return [
            VectorEmbedding(
                embeddings=vector,
                metadata={
                    "elapsed_ms": frame.elapsed_ms,
                    "frame_index": frame.frame_index,
                    "video_path": frame.video_path,
                    "timestamp": frame.timestamp
                }) 
            for vector, frame in zip(embeddings_numpy, frames)
        ]