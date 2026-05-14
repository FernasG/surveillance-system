import time, os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from guard.pipeline.acquisition.record import record_segmented

class RecordSegmentHandler(FileSystemEventHandler):
    def on_closed(self, event):
        if not event.is_directory and event.src_path.endswith(".mp4"):
            print(f"--- evento recebido hereeeeeeee: {event.src_path} ---")

            self.process_new_video(event.src_path)

    def process_new_video(self, path):
        # call processing pipeline
        pass

if __name__ == "__main__":
    
    current_file = Path(__file__).resolve()

    path_to_watch = os.path.join(current_file.parent.parent, "videos")

    event_handler = RecordSegmentHandler()
    observer = Observer()
    observer.schedule(event_handler, path_to_watch, recursive=False)
    observer.start()

    process = record_segmented()

    try:
        print("gravação iniciada")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        process.terminate() # Para o FFmpeg
        observer.stop()
    
    observer.join()