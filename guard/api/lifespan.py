from contextlib import asynccontextmanager
from fastapi import FastAPI

from guard.core.entities import Settings
from guard.infrastructure.di.container import ApplicationContainer

settings = Settings()

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    container = ApplicationContainer(settings)
    
    await container.initialize()
    
    yield {
        "retrieval_service": container.retrieval_service,
        "analytics_service": container.analytics_service,
        "auth_service": container.auth_service
    }

    await container.shutdown()