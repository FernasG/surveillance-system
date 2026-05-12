import chromadb
from chromadb.config import Settings
from guard.core.entities import VectorEmbedding
from guard.core.interfaces import VectorStoreInterface

class ChromaDBStore(VectorStoreInterface):
    def __init__(self, host: str = "chromadb", port: int = 8000):
        super().__init__()

        self.client = chromadb.HttpClient(host=host, port=port, settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name="video_surveillance")

    def save(self, embedding: VectorEmbedding, doc_id: str) -> bool:
        self.collection.add(
            embeddings=[embedding.embeddings.tolist()],
            metadatas=[embedding.metadata],
            ids=[doc_id]
        )

    def search(self, query_vector: list[float], top_k: int = 5) -> list[dict]:
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )

        return results

    def save_batch(self, embeddings: list[VectorEmbedding], doc_id: str) -> bool:
        list_of_vectors = [emp.embeddings.tolist() for emp in embeddings]
        list_of_metadatas = [emp.metadata for emp in embeddings]
        list_of_ids = [f"{doc_id}_{i}" for i in range(len(embeddings))]

        self.collection.add(
            embeddings=list_of_vectors,
            metadatas=list_of_metadatas,
            ids=list_of_ids
        )
        
        return True