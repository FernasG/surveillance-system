import json
import time
import redis
from loguru import logger

class RedisEventPublisher:
    def __init__(self, redis_client: redis.Redis, queue_name: str):
        self._redis_client = redis_client
        self._queue_name = queue_name

    def publish_event_closed(self, event_id: str, video_path: str, elapsed_ms: int) -> None:
        payload = {
            "event_id": event_id,
            "video_path": video_path,
            "elapsed_ms": elapsed_ms,
            "closed_at": time.time(),
        }

        try:
            self._redis_client.lpush(self._queue_name, json.dumps(payload))
        except Exception:
            logger.bind(event_id=event_id).exception("Failed to publish EVENT_CLOSED message")
