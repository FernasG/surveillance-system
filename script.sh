#!/bin/bash

ffmpeg \
    -loglevel error \
	-f v4l2 \
	-input_format mjpeg \
	-video_size 1280x720 \
	-framerate 30 \
	-i /dev/video0 \
	-f alsa \
	-i plughw:CARD=WEBCAM,DEV=0 \
	-t 30 \
	-c:v libx264 \
	-preset veryfast \
	-pix_fmt yuv420p \
	-r 30 \
	-c:a aac \
	output.mp4