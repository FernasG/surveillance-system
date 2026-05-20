import logging
import sys
from loguru import logger
from contextvars import ContextVar

REQUEST_ID_CTX: ContextVar[str] = ContextVar("request_id", default="nosync")

class InterceptHandler(logging.Handler):
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = sys._getframe(6)
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back

        logger.opt(depth=6, exception=record.exc_info).log(level, record.getMessage())

def setup_logging(json_format: bool = False):
    logger.remove()

    def inject_request_id(record):
        logger.configure(patcher=inject_request_id)
        record["extra"]["request_id"] = REQUEST_ID_CTX.get()

    if json_format:
        logger.add(sys.stdout, serialize=True, level="INFO")
    else:
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<magenta>[{extra[request_id]}]</magenta> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        logger.configure(patcher=inject_request_id)
        logger.add(sys.stdout, format=log_format, colorize=True, level="INFO")

    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    for logger_name in ("uvicorn", "uvicorn.asgi", "uvicorn.access", "uvicorn.error"):
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler()]
        logging_logger.propagate = False