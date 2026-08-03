import asyncio
import signal

from loguru import logger

from guard.core.entities import Settings
from guard.infrastructure.logging.logger_config import setup_logging
from guard.infrastructure.messaging.search_priority_listener import SearchPriorityListener
from guard.worker.enrichment_container import EnrichmentWorkerContainer

LISTENER_RESTART_DELAY_S = 5

async def run_listener_supervised(listener: SearchPriorityListener) -> None:
    while True:
        try:
            await listener.start()
            logger.warning("Search priority listener stream ended; restarting")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Search priority listener crashed; restarting")

        await asyncio.sleep(LISTENER_RESTART_DELAY_S)

async def main():
    settings = Settings()

    setup_logging(json_format=(settings.env == "production"))

    container = EnrichmentWorkerContainer(settings)
    await container.initialize()

    loop = asyncio.get_running_loop()

    listener_task = asyncio.create_task(run_listener_supervised(container.priority_listener))
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
