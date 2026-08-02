import asyncio
import signal

from guard.core.entities import Settings
from guard.infrastructure.logging.logger_config import setup_logging
from guard.worker.description_container import DescriptionWorkerContainer

async def main():
    settings = Settings()

    setup_logging(json_format=(settings.env == "production"))

    container = DescriptionWorkerContainer(settings)
    await container.initialize()

    loop = asyncio.get_running_loop()

    listener_task = asyncio.create_task(container.priority_listener.start())
    worker_task = asyncio.create_task(container.queue_worker.start(settings.event_queue_name))

    def _shutdown():
        worker_task.cancel()
        listener_task.cancel()

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, _shutdown)

    try:
        await worker_task
    finally:
        listener_task.cancel()
        await asyncio.gather(listener_task, return_exceptions=True)
        await container.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
