from guard.core.interfaces import VectorizerInterface, VectorStoreInterface

class RetrievalService:
    def __init__(self, vectorizer: VectorizerInterface, store: VectorStoreInterface):
        self.vectorizer = vectorizer
        self.store = store

    def search_by_text(self, text: str, top_k: int = 5):
        query_vector = self.vectorizer.encode_text(text)
        results = self.store.search(query_vector, top_k=top_k)
        
        return results