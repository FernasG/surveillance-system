import chromadb
from uuid import uuid4
from chromadb.config import Settings
from guard.core.entities import VectorEmbedding
from guard.core.interfaces import VectorStoreInterface

class ChromaDBStore(VectorStoreInterface):
    def __init__(self, host: str = "chromadb", port: int = 8000):
        super().__init__()

        self.client = chromadb.HttpClient(host=host, port=port, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name="video_surveillance")

    def save(self, embedding: VectorEmbedding) -> bool:
        self.collection.add(
            embeddings=[embedding.embeddings.tolist()],
            metadatas=[embedding.metadata],
            ids=[str(uuid4())]
        )

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        return results

    def save_batch(self, embeddings: list[VectorEmbedding]) -> bool:
        vectors_list = [emp.embeddings.tolist() for emp in embeddings]
        metadatas_list = [emp.metadata for emp in embeddings]
        ids_list = [str(uuid4()) for _ in range(len(embeddings))]

        self.collection.add(
            embeddings=vectors_list,
            metadatas=metadatas_list,
            ids=ids_list
        )
        
        return True