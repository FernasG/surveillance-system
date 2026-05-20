from contextlib import asynccontextmanager
from fastapi import FastAPI

from guard.core.entities import Settings
from guard.infrastructure.container import ApplicationContainer

settings = Settings()

@asynccontextmanager
async def app_lifespan(app: FastAPI):
    container = ApplicationContainer(settings)
    
    await container.initialize()
    
    yield {"retrieval_service": container.retrieval_service}

    await container.shutdown()