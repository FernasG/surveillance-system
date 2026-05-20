#!/bin/bash

file_path=$(pwd)
fragment_duration_s=15
framerate=30

ffmpeg \
    -loglevel error \
    -f v4l2 \
    -input_format mjpeg \
    -video_size 1280x720 \
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
    "$file_path/videos/cam1_%s.mp4"
