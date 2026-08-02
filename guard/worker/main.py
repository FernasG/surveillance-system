import asyncio
import signal

from guard.core.entities import Settings
from guard.infrastructure.logging.logger_config import setup_logging
from guard.worker.container import WorkerContainer

async def main():
    settings = Settings()

    setup_logging(json_format=(settings.env == "production"))

    container = WorkerContainer(settings)
    await container.initialize()

    loop = asyncio.get_running_loop()
    worker_task = asyncio.create_task(container.queue_worker.start(settings.redis_queue_name))

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, worker_task.cancel)

    try:
        await worker_task
    finally:
        await container.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
