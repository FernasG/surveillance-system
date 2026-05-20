import redis, time, json, os, logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class VideoSegmentHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

        redis_host = os.environ.get("REDIS_HOST", "redis")
        redis_port = os.environ.get("REDIS_PORT", "6379")
        self.queue_name = os.environ.get("REDIS_QUEUE_NAME", "video_queue")

        self._redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)

    def on_closed(self, event):
        if not event.is_directory and event.src_path.endswith(".mp4"):
            logging.info(f"[WATCHER] New segment completed: {event.src_path}")
            file_path = event.src_path
            file_name = os.path.basename(file_path)
            
            try:
                parts = file_name.split('_')
                timestamp = int(parts[1]) if len(parts) > 1 else int(time.time())
            except (ValueError, IndexError):
                timestamp = int(time.time())

            data = {
                "timestamp": timestamp,
                "video_path": file_path,
                "created_at": time.time()
            }

            self._redis_client.lpush(self.queue_name, json.dumps(data))
            logging.info(f"[WATCHER] Event sent to queue: {file_name}")


if __name__ == "__main__":
    path_to_watch = os.environ.get("VIDEOS_DIR", "/app/videos")
    
    event_handler = VideoSegmentHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()

    try:
        logging.info("[WATCHER] Monitoring and recording system started")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()