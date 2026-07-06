from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from guard.api.lifespan import app_lifespan
from guard.api.middlewares import RequestIdMiddleware
from guard.infrastructure.logging.logger_config import setup_logging
from guard.core.entities import Settings
from guard.api.routers.auth_router import router as auth_router
from guard.api.routers.video_router import router as video_router
from guard.api.routers.search_router import router as search_router

settings = Settings()

IS_PRODUCTION = settings.env == "production"

setup_logging(json_format=IS_PRODUCTION)

app = FastAPI(title="Pi Guard", lifespan=app_lifespan)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(video_router)
app.include_router(search_router)

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "OK"}
