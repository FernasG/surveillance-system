import ffmpeg
import os
from guard.core.interfaces import CameraDriver

class LogitechUSBDriver(CameraDriver):
    def __init__(self, device_path="/dev/video0"):
        self.device_path = device_path
        self.process = None

    def start_recording(self, segment_time: int, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        file_template = os.path.join(folder_path, "cam1_%s.mp4")

        v_in = ffmpeg.input(
            self.device_path,
            f="v4l2",
            input_format="mjpeg",
            video_size="1280x720",
            framerate="30"
        )

        self.process = (
            ffmpeg.output(
                v_in, file_template,
                vcodec="libx264", preset="veryfast", pix_fmt="yuv420p",
                acodec="aac", f="segment", segment_time=segment_time,
                reset_timestamps=1, segment_format="mp4", strftime=1
            )
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )

    def stop_recording(self):
        if self.process:
            self.process.communicate(input=b"q")