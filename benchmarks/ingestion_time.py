"""Metric 2 - end-to-end ingestion time of a single video through both pipeline stages.

Black-box measurement, no production code is touched:

  T0  Watcher detection   -> `created_at` written by the watcher into the Redis message
  T1  Stage 1 finished    -> message ACKed (removed) from `<video_queue>:processing`
                             (MOG2 sampling + CLIP embeddings + ChromaDB save_batch done)
  T2  Stage 2 finished    -> no frame of this video left with description == "pending"
                             in ChromaDB (YOLO + VLM description written for every event)

The preprocessing worker only closes the last event of a video when the next segment
arrives or on shutdown, so after T1 the script restarts `guard-worker` (SIGTERM ->
close_open_event) to publish it. Use --flush-mode none to skip this.

Usage:
    python benchmarks/ingestion_time.py --video cam1_1783613349.mp4
"""
import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import redis
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_COLLECTION_PATH = "/api/v2/tenants/default_tenant/databases/default_database/collections"
COLLECTION_NAME = "video_surveillance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end ingestion time benchmark")
    parser.add_argument("--video", required=True, help="source .mp4 (named <cam>_<epoch>.mp4)")
    parser.add_argument("--videos-dir", default=str(REPO_ROOT / "videos"), help="host dir watched by the watcher")
    parser.add_argument("--container-videos-dir", default=None, help="same dir inside containers (default VIDEOS_DIR from .env)")
    parser.add_argument("--env-file", default=str(REPO_ROOT / ".env"))
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--chroma-url", default="http://localhost:8000")
    parser.add_argument("--flush-mode", choices=["restart", "none"], default="restart")
    parser.add_argument("--worker-container", default="guard-worker")
    parser.add_argument("--poll", type=float, default=2.0, help="ChromaDB poll interval (s)")
    parser.add_argument("--grace", type=float, default=120.0, help="seconds with empty event queues before giving up on pending events")
    parser.add_argument("--timeout", type=float, default=6 * 3600.0)
    parser.add_argument("--out", default="results/ingestion.csv")

    return parser.parse_args()


def load_env(path: str) -> dict:
    env = {}

    if not Path(path).exists():
        return env

    for line in Path(path).read_text().splitlines():
        line = line.strip()

        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip('"').strip("'")

    return env


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


class Chroma:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

        response = requests.get(f"{self.base_url}{CHROMA_COLLECTION_PATH}/{COLLECTION_NAME}", timeout=30)
        response.raise_for_status()

        self.collection_id = response.json()["id"]

    def count(self, where: dict) -> int:
        response = requests.post(
            f"{self.base_url}{CHROMA_COLLECTION_PATH}/{self.collection_id}/get",
            json={"where": where, "include": []},
            timeout=120,
        )
        response.raise_for_status()

        return len(response.json()["ids"])


def find_message(client: redis.Redis, queues: list[str], video_path: str) -> dict | None:
    for queue in queues:
        for raw in client.lrange(queue, 0, -1):
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if message.get("video_path") == video_path:
                return message

    return None


def video_duration_s(path: Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None

    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )

    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))

        if is_new:
            writer.writeheader()

        writer.writerow(row)


def main() -> int:
    args = parse_args()
    env = load_env(args.env_file)

    video_queue = env.get("REDIS_QUEUE_NAME", "video_queue")
    event_queue = env.get("EVENT_QUEUE_NAME", "event_queue")
    container_dir = (args.container_videos_dir or env.get("VIDEOS_DIR", "/app/videos")).rstrip("/")

    source = Path(args.video).resolve()
    target = Path(args.videos_dir).resolve() / source.name
    container_path = f"{container_dir}/{source.name}"

    if not source.is_file():
        log(f"Source video not found: {source}")
        return 1

    if target.exists():
        log(f"{target} already exists; the watcher only fires on new files. Remove it (and its vectors) first.")
        return 1

    client = redis.Redis(host=args.redis_host, port=args.redis_port, decode_responses=True)
    chroma = Chroma(args.chroma_url)

    video_queues = [video_queue, f"{video_queue}:processing"]
    event_queues = [event_queue, f"{event_queue}:processing"]
    busy = {q: client.llen(q) for q in video_queues + event_queues if client.llen(q)}

    if busy:
        log(f"Pipeline is not idle, wait until these queues drain: {busy}")
        return 1

    duration = video_duration_s(source)
    log(f"Video: {source.name} ({duration / 60:.1f} min)" if duration else f"Video: {source.name}")

    # --- Trigger: the watcher fires on_closed when the copy finishes ---
    copy_start = time.time()
    shutil.copyfile(source, target)
    copy_end = time.time()
    log(f"Copied into {target.parent} in {copy_end - copy_start:.1f}s, waiting for watcher")

    message = None
    deadline = time.time() + 60

    while message is None and time.time() < deadline:
        message = find_message(client, video_queues, container_path)
        if message is None:
            time.sleep(0.05)

    if message is None:
        log(f"Watcher did not enqueue {container_path} within 60s (check VIDEOS_DIR / watcher logs)")
        return 1

    t0 = float(message["created_at"])
    log(f"T0 watcher detection at {datetime.fromtimestamp(t0).isoformat()}")

    # --- Stage 1: wait until the preprocessing worker ACKs the message ---
    start = time.time()
    last_progress = 0.0

    while find_message(client, video_queues, container_path) is not None:
        if time.time() - start > args.timeout:
            log("Timeout waiting for stage 1")
            return 1

        if time.time() - last_progress >= 30:
            frames = chroma.count({"video_path": container_path})
            log(f"Stage 1 running: {frames} frames indexed so far")
            last_progress = time.time()

        time.sleep(0.5)

    t1 = time.time()
    log(f"T1 stage 1 done: {t1 - t0:.1f}s after T0")

    if args.flush_mode == "restart":
        log(f"Restarting {args.worker_container} to flush the last open event")
        subprocess.run(["docker", "restart", "-t", "30", args.worker_container], check=True, capture_output=True)

    # --- Stage 2: wait until every frame of this video has a description ---
    pending_where = {"$and": [{"video_path": container_path}, {"description": "pending"}]}
    idle_since = None
    last_progress = 0.0
    pending = chroma.count(pending_where)

    while pending > 0:
        now = time.time()

        if now - t0 > args.timeout:
            log("Timeout waiting for stage 2")
            break

        queued = sum(client.llen(q) for q in event_queues)

        if queued == 0:
            idle_since = idle_since or now

            if now - idle_since > args.grace:
                log(f"Event queues empty for {args.grace:.0f}s with {pending} frames still pending; treating as dropped")
                break
        else:
            idle_since = None

        if now - last_progress >= 30:
            log(f"Stage 2 running: {pending} frames pending, {queued} events queued")
            last_progress = now

        time.sleep(args.poll)
        pending = chroma.count(pending_where)

    t2 = time.time()

    frames = chroma.count({"video_path": container_path})
    events = chroma.count({"$and": [{"video_path": container_path}, {"is_keyframe": True}]})
    dropped_events = chroma.count({"$and": [{"video_path": container_path}, {"description": "pending"}, {"is_keyframe": True}]})

    total = t2 - t0
    row = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "video": source.name,
        "video_duration_s": round(duration, 1) if duration else "",
        "copy_s": round(copy_end - copy_start, 2),
        "t0": round(t0, 3),
        "t1": round(t1, 3),
        "t2": round(t2, 3),
        "stage1_s": round(t1 - t0, 1),
        "stage2_tail_s": round(t2 - t1, 1),
        "total_s": round(total, 1),
        "frames": frames,
        "events": events,
        "dropped_events": dropped_events,
        "realtime_factor": round(total / duration, 3) if duration else "",
        "flush_mode": args.flush_mode,
    }
    append_row(Path(args.out), row)

    print("\n=== Ingestion summary ===")
    for key, value in row.items():
        print(f"{key:>17}: {value}")
    print(f"\nTotal: {total / 60:.1f} min ({total / 3600:.2f} h). Row appended to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
