from guard.core.interfaces import CameraDriver

class AcquisitionService:
    def __init__(self, camera_driver: CameraDriver):
        self.camera_driver = camera_driver

    def start(self, segment_time: int = 10, path: str = "videos"):
        self.camera_driver.start_recording(segment_time, path)

    def stop(self):
        self.camera_driver.stop_recording()