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

    def search(self, query_vector: list[float], top_events: int = 5, fetch_multiplier: int = 10) -> list[dict]:
        fetch_count = top_events * fetch_multiplier
        
        raw_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=fetch_count
        )

        if not raw_results["ids"][0]:
            return []

        unique_events = []
        seen_event_ids = set()

        ids = raw_results["ids"][0]
        distances = raw_results["distances"][0]
        metadatas = raw_results["metadatas"][0]
        
        documents = raw_results["documents"][0] if raw_results.get("documents") else [""] * len(ids)

        for item_id, distance, metadata, document in zip(ids, distances, metadatas, documents):
            event_id = metadata.get("event_id")

            if not event_id:
                event_id = item_id

            if event_id not in seen_event_ids:
                seen_event_ids.add(event_id)
                
                unique_events.append({
                    "frame_id": item_id,
                    "event_id": event_id,
                    "distance": distance,
                    "metadata": metadata,
                    "description": document
                })

            if len(unique_events) >= top_events:
                break

        return unique_events

    def save_batch(self, embeddings: list[VectorEmbedding]) -> bool:
        vectors_list = [emp.embeddings.tolist() for emp in embeddings]
        metadatas_list = [emp.metadata for emp in embeddings]
        ids_list = [str(uuid4()) for _ in range(len(embeddings))]

        documents_list = [
            emp.metadata.get("description", "") if emp.metadata else "" 
            for emp in embeddings
        ]

        self.collection.add(
            embeddings=vectors_list,
            metadatas=metadatas_list,
            documents=documents_list,
            ids=ids_list
        )
        
        return True