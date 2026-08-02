from guard.core.entities import Settings
from guard.infrastructure.models.gemma_vlm import GemmaVLM
from guard.infrastructure.models.clip_vectorizer import CLIPVectorizer
from guard.infrastructure.database.chromadb_store import ChromaDBStore
from guard.infrastructure.database.sqlite_analytics_store import SqliteAnalyticsStore
from guard.infrastructure.models.utils.prompt_manager import PromptManager
from guard.infrastructure.database.sqlite_auth_store import SqliteAuthStore
from guard.infrastructure.database.sqlite_store import SqliteDatabase

from guard.core.services.retrieval_service import RetrievalService
from guard.core.services.auth_service import AuthService
from guard.core.services.analytics_service import AnalyticsService

class ApplicationContainer:
    def __init__(self, settings: Settings):
        self.settings = settings

        self.vectorizer = None
        self.analytics_service = None
        self.auth_service = None
        self.retrieval_service = None

    async def initialize(self):
        self.vectorizer = CLIPVectorizer()
        prompt_manager = PromptManager()
        gemma_vlm = GemmaVLM()

        store = ChromaDBStore(host=self.settings.database_host, port=self.settings.database_port)

        database = SqliteDatabase()
        auth_repo = SqliteAuthStore(database)
        self.auth_service = AuthService(auth_repo)
        await self.auth_service.initialize_admin()

        analytics_repo = SqliteAnalyticsStore(database)
        self.analytics_service = AnalyticsService(analytics_repo)

        self.retrieval_service = RetrievalService(vectorizer=self.vectorizer, store=store, vlm=gemma_vlm, prompt_manager=prompt_manager)

    async def shutdown(self):
        if self.vectorizer:
            del self.vectorizer
