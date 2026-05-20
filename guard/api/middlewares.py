import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from loguru import logger
from guard.infrastructure.logging.logger_config import REQUEST_ID_CTX

class RequestIdMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = uuid.uuid4().hex
        
        token = REQUEST_ID_CTX.set(request_id)
        
        logger.info(f"Receiving request: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            response.headers["X-Request-ID"] = request_id
            
            logger.info(f"Responded with status {response.status_code}")
            return response
        except Exception as e:
            logger.exception("Critical error during request processing")
            raise e
        finally:
            REQUEST_ID_CTX.reset(token)