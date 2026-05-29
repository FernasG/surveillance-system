import io
import grpc
import logging
import threading
import ml_server_pb2
import ml_server_pb2_grpc
from PIL import Image
from concurrent import futures
from models.language_model import LanguageModel
from models.embedding_model import EmbeddingModel

class MLServerServicer(ml_server_pb2_grpc.MLServerServicer):
    def __init__(self):
        self.models_ready = threading.Event()
        self.language_model = LanguageModel()
        self.embedding_model = EmbeddingModel()

        threading.Thread(target=self._load_models, daemon=True).start()

    def GenerateText(self, request, context):
        if not self.models_ready.is_set():
            logging.info("Request received but models are still loading. Waiting...")
            self.models_ready.wait()

        logging.info(f"LLM Prompt received: '{request.prompt}'")

        pil_images = [Image.open(io.BytesIO(img)).convert("RGB") for img in request.images]

        content = [{"type": "image"} for _ in pil_images]
        content.append({"type": "text", "text": request.prompt})
        messages = [{"role": "user", "content": content}]
        
        text = self.language_model.generate(messages=messages, images=pil_images)

        return ml_server_pb2.GenerateTextResponse(text=text)

    def EncodeText(self, request, context):
        if not self.models_ready.is_set():
            logging.info("Request received but models are still loading. Waiting...")
            self.models_ready.wait()

        logging.info(f"Embedding request received for text length: {len(request.text)}")
        
        embedding = self.embedding_model.encode_text(request.text)

        return ml_server_pb2.EncodeTextResponse(embedding=embedding)
    
    def EncodeImage(self, request, context):
        if not self.models_ready.is_set():
            logging.info("Request received but models are still loading. Waiting...")
            self.models_ready.wait()

        logging.info("Embedding request received for an image")
        
        image_bytes = request.image_data
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        embedding = self.embedding_model.encode_image(pil_img)

        return ml_server_pb2.EncodeImageResponse(embedding=embedding)
    
    def EncodeBatchImages(self, request, context):
        if not self.models_ready.is_set():
            logging.info("Request received but models are still loading. Waiting...")
            self.models_ready.wait()

        logging.info(f"Embedding request received for a batch of images with size: {len(request.images_data)}")

        images_bytes = request.images_data
        pil_imgs = [Image.open(io.BytesIO(image_byte)).convert("RGB") for image_byte in images_bytes]
        numpy_embeddings = self.embedding_model.encode_batch_images(pil_imgs)

        proto_embeddings = [
            ml_server_pb2.EncodeImageResponse(embedding=emb.tolist())
            for emb in numpy_embeddings
        ]

        return ml_server_pb2.EncodeBatchImagesResponse(embeddings=proto_embeddings)
    
    def _load_models(self):
        logging.info("Loading ML models into memory...")
        
        # self.embedding_model.initialize()
        self.language_model.initialize()

        logging.info("Models loaded successfully!")

        self.models_ready.set()

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    ml_server_pb2_grpc.add_MLServerServicer_to_server(MLServerServicer(), server)
    server.add_insecure_port('[::]:50051')
    
    logging.info("gRPC ML Server running on port 50051...")
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    serve()