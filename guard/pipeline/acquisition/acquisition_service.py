import redis, time
from watchdog.observers import Observer
from ....watcher.watcher import VideoSegmentHandler

class AcquisitionService:
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client

    def start_watcher(self, folder_path: str):
        event_handler = VideoSegmentHandler(self.redis_client)
        observer = Observer()
        observer.schedule(event_handler, folder_path, recursive=False)
        
        observer.start()
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        
        observer.join()
