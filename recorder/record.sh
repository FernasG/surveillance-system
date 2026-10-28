#!/bin/sh
# Records each go2rtc stream into fixed-length MP4 segments named
# <camera>_<epoch>.mp4, which the watcher picks up from the videos directory.
set -eu

: "${RTSP_URL:?RTSP_URL must be set}"
cameras="${RECORD_CAMERAS:-cam1}"
segment_seconds="${RECORD_SEGMENT_SECONDS:-3600}"
videos_dir="${VIDEOS_DIR:-/app/videos}"

record() {
    camera="$1"

    while true; do
        ffmpeg \
            -loglevel error \
            -rtsp_transport tcp \
            -i "$RTSP_URL/$camera" \
            -map 0:v \
            -c copy \
            -f segment \
            -segment_time "$segment_seconds" \
            -segment_atclocktime 1 \
            -reset_timestamps 1 \
            -segment_format mp4 \
            -strftime 1 \
            "$videos_dir/${camera}_%s.mp4" || true

        echo "[RECORDER] $camera: stream ended, reconnecting in 5 s"
        sleep 5
    done
}

# Forward SIGTERM to every ffmpeg so the open segment is finalized on stop.
trap 'trap - TERM INT; kill -TERM 0; wait; exit 0' TERM INT

for camera in $cameras; do
    echo "[RECORDER] Recording $camera"
    record "$camera" &
done

wait
