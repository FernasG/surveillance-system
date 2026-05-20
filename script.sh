#!/bin/bash

folder_path=$(pwd)
resolution=1280x720
fragment_duration_s=15
framerate=30

ffmpeg \
    -loglevel error \
    -f v4l2 \
    -input_format mjpeg \
    -video_size "$resolution" \
    -framerate "$framerate" \
    -i /dev/video0 \
    -c:v libx264 \
    -preset veryfast \
    -pix_fmt yuv420p \
    -f segment \
    -segment_time "$fragment_duration_s" \
    -reset_timestamps 1 \
    -segment_format mp4 \
    -strftime 1 \
    "$folder_path/videos/cam1_%s.mp4"
