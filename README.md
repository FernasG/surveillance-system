# Surveillance System (Pi Guard)

Pi Guard is a self-hosted, camera-agnostic surveillance system built to run on modest hardware (originally targeting a Raspberry Pi-class device). It continuously records video, detects and clusters motion into discrete **events** in real time, and asynchronously enriches each event with an AI-generated natural-language description and object tags — so you can later search your footage with plain-English queries like *"person carrying a package at the front door"* instead of scrubbing through hours of recordings.

The system is split into two decoupled processing stages so that **searching your footage never has to wait for the AI captioning pipeline**: an active search request preempts in-flight description generation, gets an immediate answer, and lets the background worker resume afterwards.

A companion web frontend is available separately at [FernasG/surveillance-manager](https://github.com/FernasG/surveillance-manager).

---

## How it works

**Stage 1 — Ingestion & Clustering** (`guard-worker`)
Video segments recorded by `ffmpeg`/the watcher are pulled off a Redis queue. Each segment is sampled once per second, background-subtracted with OpenCV's MOG2 to find motion regions, and embedded with CLIP. Frames are clustered into events online using a combination of time proximity, embedding similarity (against an EMA centroid), and motion bounding-box IoU — no second pass or global clustering step is needed. Closed events are published to a second queue for Stage 2.

**Stage 2 — Async Description & Detection** (`guard-description-worker`)
For each closed event, the worker extracts its most representative frame (the one with the largest motion area), runs YOLO object detection on it, and asks a VLM (served locally via `llama.cpp`) to describe what's happening. The result is written back into the event's vector store entry.

**Search preemption**
Because Stage 2's VLM calls can take a while on CPU/edge hardware, a search request from the API signals the description worker to cancel its current VLM call and step aside until the search completes, so interactive queries stay fast even while the system is busy captioning the backlog.

**API**
A FastAPI service exposes authentication, video listing/playback/thumbnails, live camera streaming, semantic search, and analytics endpoints, all backed by the same ChromaDB/SQLite stores the workers write to.

---

## Tech stack

| Concern                   | Technology                                       |
|---------------------------|--------------------------------------------------|
| API                       | FastAPI + Uvicorn                                |
| Motion detection          | OpenCV (MOG2 background subtraction)             |
| Image/text embeddings     | CLIP (served via a small dedicated `clip-api`)   |
| Object detection          | YOLO (Ultralytics)                               |
| Vision-language captioning| `llama.cpp` (Gemma / LFM / Qwen GGUF models)     |
| Vector store              | ChromaDB                                         |
| Relational store          | SQLite (WAL mode, shared across API/workers)     |
| Messaging / queues        | Redis                                            |
| Segment recording         | `ffmpeg` (V4L2 capture) + `watchdog`             |
| Auth                      | JWT (PyJWT) + bcrypt                             |

---

## Project layout

```
guard/
├── api/                # FastAPI app: routers, middleware, lifespan/DI wiring
├── core/                # Domain: entities, interfaces (ports), services
├── infrastructure/      # Adapters: ChromaDB/SQLite/Redis, models (CLIP/YOLO/VLM), DI containers
├── pipeline/            # Acquisition, preprocessing, inference, description use cases
└── worker/               # Entry points for the two background workers

clip-api/                # Standalone CLIP embedding microservice
watcher/                 # Watches videos/ for finished segments and enqueues them
llama.cpp/                # Dockerfile + scripts to serve local VLMs
```

The codebase follows a clean-architecture split: `core` has no dependency on `infrastructure`, `infrastructure` implements the interfaces `core` declares, and `api`/`worker` wire concrete adapters into services via dependency injection containers.

---

## Prerequisites

- Docker and Docker Compose
- A USB camera exposed at `/dev/video0` (for live streaming and recording)
- Enough disk space for recorded segments and downloaded GGUF model weights (several GB)
- A GPU is optional but recommended for the `llama.cpp` and CLIP/YOLO services if you want faster inference

---

## Getting started

1. **Clone the repository and copy the environment file**

   ```bash
   git clone <this-repo-url>
   cd surveillance-system
   cp .env.example .env
   ```

2. **Fill in `.env`.** See [Configuration](#configuration) below for what each variable does.

3. **Start the recording pipeline.** Segments are produced by `script.sh`, which captures from `/dev/video0` with `ffmpeg` and writes 15-second `.mp4` fragments into `videos/`:

   ```bash
   ./script.sh
   ```

   (Run this on the host, or adapt it into your own capture setup — anything that writes finished `.mp4` segments into `videos/` will work, since the `watcher` service picks them up from there.)

4. **Build and start everything else:**

   ```bash
   make up-build
   ```

   This brings up, in dependency order: `redis`, `chromadb`, `clip-api`, the `llama.cpp`-based model servers (downloading GGUF weights on first run via `llama-downloader`), the `watcher`, both `guard` workers, and the `pi-guard` API.

5. **Check it's alive:**

   ```bash
   curl http://localhost:3000/healthcheck
   ```

6. **Log in.** An admin user is seeded automatically on first boot with username `admin` / password `admin` — change this immediately (see [Authentication](#authentication)).

7. **(Optional) Run the frontend.** Clone and run [FernasG/surveillance-manager](https://github.com/FernasG/surveillance-manager) separately and point it at this API (default `http://localhost:3000`).

### Useful Make targets

```bash
make build       # docker compose build
make up          # docker compose up
make up-build    # docker compose up --build
make down        # docker compose down
make logs        # follow logs from all services
make sh:guard    # shell into the pi-guard API container
make sh:watcher  # shell into the watcher container
```

---

## Configuration

All configuration is read from `.env` (see `.env.example` for the full list of keys). The important ones:

| Variable | Purpose |
|---|---|
| `ENV` | `production` enables JSON structured logging; anything else uses human-readable colored logs |
| `DATABASE_HOST` / `DATABASE_PORT` | ChromaDB connection |
| `CLIP_SERVER_URL` | Base URL of the `clip-api` service |
| `VIDEOS_DIR` | Path (inside the container) where recorded segments live |
| `REDIS_HOST` / `REDIS_PORT` | Redis connection shared by all workers and the API |
| `REDIS_QUEUE_NAME` | Queue the watcher publishes new video segments to (Stage 1 input) |
| `EVENT_QUEUE_NAME` | Queue Stage 1 publishes closed events to (Stage 2 input) |
| `GENERATION_MODEL` / `GENERATION_SERVER_URL` | Model name/URL used for search-time VLM re-ranking |
| `IMAGE_MODEL` / `IMAGE_SERVER_URL` | Model name/URL used by Stage 2 for event captioning |
| `SECRET_KEY` / `ALGORITHM` | JWT signing configuration — set `SECRET_KEY` to a long random value |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT access token lifetime |

The `llama.cpp`-based model servers (`gemma-api`, `liquid-api`, and the disabled-by-default `qwen-api`) are configured directly in `docker-compose.yml` and `llama.cpp/scripts/`; swap the `start_*.sh` script referenced by a service's `entrypoint` to change which model it serves, and add the corresponding download step to `llama.cpp/scripts/download_models.sh`.

---

## Authentication

- The first API boot seeds an `admin` user (`admin` / `admin`, `is_admin=true`) if none exists.
- `POST /login` exchanges credentials for a JWT bearer token.
- `POST /register` lets an existing admin create new (non-admin) users.
- Protected routes accept the token either as a standard `Authorization: Bearer <token>` header or, for endpoints consumed directly by `<video>`/`<img>` tags, as a `?token=` query parameter.

**Change the default admin password immediately after first boot.**

---

## API overview

All routes below (except `/healthcheck`, `/login`, `/register`) require a valid bearer token.

| Method | Route | Description |
|---|---|---|
| `GET` | `/healthcheck` | Liveness check |
| `POST` | `/login` | Authenticate and receive a JWT |
| `POST` | `/register` | Create a new user (admin only) |
| `GET` | `/videos/` | Paginated list of recorded segments, optionally filtered by date range |
| `GET` | `/videos/live` | MJPEG live stream from the local USB camera |
| `GET` | `/videos/{video_name}/playback` | Range-request video streaming for playback |
| `GET` | `/videos/{video_name}/thumbnail` | JPEG thumbnail for a segment |
| `POST` | `/query` | Semantic search over captioned events (natural-language text query) |
| `GET` | `/metrics/daily` | Event/object analytics for a date or date range (`start_date`/`end_date`, default: today) |
| `GET` | `/metrics/overall` | All-time event/object analytics |

Interactive OpenAPI docs are available at `http://localhost:3000/docs` once the API is running.

---

## Development notes

- Each `guard/*` service (`pi-guard`, `guard-worker`, `guard-description-worker`) shares the same Docker image and codebase, differentiated only by the command they run — see `docker-compose.yml`.
- `guard/infrastructure/di/container.py` and `guard/worker/*container.py` wire concrete adapters (ChromaDB, SQLite, Redis, CLIP, VLMs) into the domain services; swap an adapter by changing what's constructed there, not by touching `core`.
- SQLite runs in WAL mode so the API (reads) and the workers (writes) can safely share the same database file across containers.
- The `guard` volume is bind-mounted into all three `guard/*` containers with `--reload` enabled on the API, so local code changes are picked up without rebuilding.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
