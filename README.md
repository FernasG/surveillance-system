# Surveillance System (Pi Guard)

Pi Guard is a self-hosted, camera-agnostic surveillance system built to run on modest hardware (originally targeting a Raspberry Pi-class device). It continuously records video, detects and clusters motion into discrete **events** in real time, and asynchronously enriches each event with an AI-generated natural-language description and object tags — so you can later search your footage with plain-English queries like *"person carrying a package at the front door"* instead of scrubbing through hours of recordings.

The system is split into two decoupled processing stages so that **searching your footage never has to wait for the AI captioning pipeline**: an active search request preempts in-flight description generation, gets an immediate answer, and lets the background worker resume afterwards.

A companion web frontend is available separately at [FernasG/surveillance-manager](https://github.com/FernasG/surveillance-manager).

---

## How it works

**Capture & recording** (`go2rtc` + `recorder`)
go2rtc runs on the Pi host and is the only process that opens the camera: it runs `rpicam-vid` and republishes the stream over RTSP. The `recorder` container reads that stream and writes one-hour `.mp4` segments into `videos/`, where the `watcher` picks them up and enqueues them in Redis.

**Stage 1 — Ingestion & Clustering** (`guard-worker`)
Each segment is sampled once per second, background-subtracted with OpenCV's MOG2 to find motion regions, and embedded with CLIP. Frames are clustered into events online using a combination of time proximity, embedding similarity (against an EMA centroid), and motion bounding-box IoU — no second pass or global clustering step is needed. Closed events are published to a second queue for Stage 2.

**Stage 2 — Async Description & Detection** (`guard-enrichment-worker`)
For each closed event, the worker extracts its most representative frame (the one with the largest motion area), runs YOLO object detection on it, and asks a VLM (served locally via `llama.cpp`) to describe what's happening. The result is written back into the event's vector store entry.

**Search preemption**
Because Stage 2's VLM calls can take a while on CPU/edge hardware, a search request from the API signals the enrichment worker to cancel its current VLM call and step aside until the search completes, so interactive queries stay fast even while the system is busy captioning the backlog.

**API**
A FastAPI service (`pi-guard`) exposes authentication, video listing/playback/thumbnails, live camera streaming, semantic search, and analytics endpoints, all backed by the same ChromaDB/SQLite stores the workers write to.

---

## Tech stack

| Concern                    | Technology                                       |
|----------------------------|--------------------------------------------------|
| API                        | FastAPI + Uvicorn                                |
| HTTPS / reverse proxy      | Caddy (local CA, no internet needed)             |
| Motion detection           | OpenCV (MOG2 background subtraction)             |
| Image/text embeddings      | CLIP (served via a small dedicated `clip-api`)   |
| Object detection           | YOLO (Ultralytics)                               |
| Vision-language captioning | `llama.cpp` (Qwen3.5-0.8B MTP, GGUF)             |
| Vector store               | ChromaDB                                         |
| Relational store           | SQLite (WAL mode, shared across API/workers)     |
| Messaging / queues         | Redis                                            |
| Camera / segment recording | `go2rtc` (RTSP) + `ffmpeg` recorder + `watchdog` |
| Auth                       | JWT (PyJWT) + bcrypt                             |

---

## Project layout

```
guard/
├── api/                 # FastAPI app: routers, middleware, lifespan/DI wiring
├── core/                # Domain: entities, interfaces (ports), services
├── infrastructure/      # Adapters: ChromaDB/SQLite/Redis, models (CLIP/YOLO/VLM), camera, DI containers
├── pipeline/            # Acquisition, preprocessing, inference, description use cases
└── worker/              # Entry points for the two background workers

caddy/                   # Caddyfile: HTTPS entry point for the API and the frontend
clip-api/                # Standalone CLIP embedding microservice
go2rtc/                  # go2rtc config + installer (runs on the Pi host, not in Docker)
llama.cpp/               # Dockerfile + scripts to serve the local VLM
recorder/                # ffmpeg container that records go2rtc streams into videos/
watcher/                 # Watches videos/ for finished segments and enqueues them
```

The codebase follows a clean-architecture split: `core` has no dependency on `infrastructure`, `infrastructure` implements the interfaces `core` declares, and `api`/`worker` wire concrete adapters into services via dependency injection containers.

---

## Environments

The same `docker-compose.yml` runs in two modes. **Production** is the default; **development** layers `docker-compose.dev.yml` on top.

|                          | Production (`make up`)                              | Development (`make dev`)                                  |
|--------------------------|-----------------------------------------------------|-----------------------------------------------------------|
| Open in the browser      | `https://<pi-address>/`                             | `http://localhost:5173/`                                  |
| Frontend calls the API at| `https://<pi-address>/api`                          | `http://localhost:3000`                                   |
| Traffic                  | HTTPS (Caddy, local CA)                             | Plain HTTP                                                |
| Published ports          | `80`, `443` (Caddy only)                            | `80`, `443`, `3000` (API), `6379`, `8000`, `8081`, `8083` |
| CORS                     | Not needed (frontend and API share one origin)      | Must list the frontend origin                             |
| API code                 | Baked into the image (rebuild to update)            | `guard/` bind-mounted, API with `--reload`                |
| go2rtc web UI            | Only on the Pi (`127.0.0.1:1984`)                   | On the network (`http://<pi-address>:1984`)               |

How a request flows in each mode:

```
Production:   browser ──HTTPS──▶ caddy:443 ─┬─ /api/* ──▶ pi-guard:3000 (internal network only)
                                            └─ /*     ──▶ surveillance-manager:5173 (MANAGER_UPSTREAM)

Development:  browser ──HTTP──▶ surveillance-manager:5173
              browser ──HTTP──▶ pi-guard:3000 (published, CORS-checked)
```

In both modes Redis, ChromaDB, `clip-api`, `qwen-api`, the workers and go2rtc's RTSP server are protected: production does not publish them at all, and RTSP always requires `RTSP_USER`/`RTSP_PASSWORD`.

> **Don't mix the modes.** In development the API runs without the `/api` prefix, so opening the frontend through Caddy while running `make dev` breaks the generated thumbnail URLs. Use `http://localhost:5173` in development and `https://<pi-address>` in production.

---

## Prerequisites

- Docker and Docker Compose
- A Raspberry Pi camera with `rpicam-apps` installed on the host (or any RTSP camera added to `go2rtc/go2rtc.yaml`)
- Enough disk space for recorded segments and downloaded GGUF model weights (several GB)
- The [surveillance-manager](https://github.com/FernasG/surveillance-manager) frontend running on the Pi (port `5173`)

---

## Setup (common to both environments)

1. **Clone and create the `.env`:**

   ```bash
   git clone <this-repo-url>
   cd surveillance-system
   cp .env.example .env
   ```

2. **Fill in the secrets** in `.env` (see [Configuration reference](#configuration-reference)):

   ```env
   SECRET_KEY=<long random value, e.g. `openssl rand -hex 32`>
   ADMIN_PASSWORD=<at least 8 characters, not "admin">
   RTSP_USER=<go2rtc user>
   RTSP_PASSWORD=<go2rtc password>
   ```

3. **Install go2rtc on the Pi host** as the `go2rtc` systemd service. It reads `RTSP_USER`/`RTSP_PASSWORD` from this `.env`:

   ```bash
   ./go2rtc/install.sh        # production: web UI only on localhost
   ./go2rtc/install.sh --dev  # development: web UI on the network (port 1984)
   ```

   Run it again with the other flag to switch modes. IP cameras can be added as extra RTSP sources in `go2rtc/go2rtc.yaml` and listed in `RECORD_CAMERAS`.

Then follow the steps for the environment you want.

---

## Production setup

1. **Set the addresses** in this project's `.env`. `<pi-address>` is how clients reach the Pi — its IP and/or its `.local` hostname:

   ```env
   SITE_ADDRESS=https://192.168.1.14, https://raspberrypi.local
   DEFAULT_SNI=192.168.1.14     # bare IP, no https://
   CORS_ORIGINS=                # empty
   ```

2. **Point the frontend at the API** in the surveillance-manager's `.env`, then recreate its container (`docker restart` does not reload the `.env`):

   ```env
   PUBLIC_API_URL=https://192.168.1.14/api
   ```

   ```bash
   docker compose up -d --force-recreate   # inside surveillance-manager/
   ```

3. **Build and start:**

   ```bash
   make up-build
   ```

4. **Export the local CA certificate** (created by Caddy on first start):

   ```bash
   make ca-cert   # writes caddy-root.crt
   openssl x509 -in caddy-root.crt -noout -fingerprint -sha256
   ```

5. **Trust the CA on every client** (laptop, phone) that opens the system. Compare the fingerprint above before importing. This is done once: the root certificate is valid for 10 years as long as `caddy_data/` is kept.

   | Client | How |
   |---|---|
   | Chrome / Brave / Edge (Linux) | `certutil -d sql:$HOME/.pki/nssdb -A -t "C,," -n "Pi Guard Local CA" -i caddy-root.crt` (`sudo apt install libnss3-tools`), or *Settings → Privacy and security → Security → Manage certificates → Authorities → Import* |
   | Firefox | *Settings → Privacy & Security → Certificates → View Certificates → Authorities → Import*, check *Trust this CA to identify websites* |
   | Linux system (`curl`, scripts) | `sudo cp caddy-root.crt /usr/local/share/ca-certificates/pi-guard-ca.crt && sudo update-ca-certificates` |
   | Android | *Settings → Security → Encryption & credentials → Install a certificate → CA certificate* |
   | Windows / macOS | Import into *Trusted Root Certification Authorities* / *Keychain Access → System*, set to *Always Trust* |

   Restart the browser afterwards. Until the CA is trusted the browser shows a "connection not secure" warning; don't bypass it with an exception, since Caddy renews the site certificate every few hours.

6. **Check it's alive and log in** as `admin` with `ADMIN_PASSWORD`:

   ```bash
   curl --cacert caddy-root.crt https://192.168.1.14/api/healthcheck
   ```

   Then open `https://192.168.1.14/`.

---

## Development setup

1. **Set the CORS origin** in this project's `.env` (the address you open the frontend from):

   ```env
   CORS_ORIGINS=http://localhost:5173
   ```

   `SITE_ADDRESS` must still be set (Caddy also runs in development), but it is not used to reach the app.

2. **Point the frontend at the API directly** in the surveillance-manager's `.env`, then recreate its container:

   ```env
   PUBLIC_API_URL=http://localhost:3000
   ```

   ```bash
   docker compose up -d --force-recreate   # inside surveillance-manager/
   ```

3. **Build and start with the development overrides:**

   ```bash
   make dev-build
   ```

4. **Open `http://localhost:5173/`** and log in as `admin` with `ADMIN_PASSWORD`. API docs: `http://localhost:3000/docs`.

**Opening the app from another machine** (e.g. a laptop while everything runs on the Pi): `localhost` only works on the Pi itself. Use the Pi's IP in both places:

```env
# surveillance-system/.env
CORS_ORIGINS=http://192.168.1.14:5173
# surveillance-manager/.env
PUBLIC_API_URL=http://192.168.1.14:3000
```

---

## Switching between environments

| From → to | Change in `surveillance-system/.env` | Change in `surveillance-manager/.env` | Run |
|---|---|---|---|
| Development → production | `CORS_ORIGINS=` | `PUBLIC_API_URL=https://<pi-address>/api` | `make down && make up-build`; recreate the manager |
| Production → development | `CORS_ORIGINS=http://localhost:5173` | `PUBLIC_API_URL=http://localhost:3000` | `make down && make dev-build`; recreate the manager |

Optionally reinstall go2rtc with or without `--dev`.

---

## Useful Make targets

```bash
make build       # docker compose build
make up          # production: docker compose up
make up-build    # production: docker compose up --build
make dev         # development: same as up, with docker-compose.dev.yml
make dev-build   # development: same as up-build, with docker-compose.dev.yml
make down        # stop everything (either mode)
make logs        # follow logs from all services
make ca-cert     # export Caddy's local root certificate to caddy-root.crt
make sh:guard    # shell into the pi-guard API container
make sh:watcher  # shell into the watcher container
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Browser says the connection is not secure | The local CA is not trusted on this client yet — see step 5 of [Production setup](#production-setup). |
| `TLS connect error ... internal error` when opening by IP | `DEFAULT_SNI` is missing or not a bare IP. Set `DEFAULT_SNI=192.168.1.14` and run `make up` again. |
| Frontend can't reach the API | `PUBLIC_API_URL` doesn't match the mode, or the manager container wasn't recreated after changing it. In development also check that `CORS_ORIGINS` contains the exact origin shown in the browser's address bar. |
| Thumbnails broken, rest works | The modes are mixed: the frontend is opened through Caddy while running `make dev`. |
| API exits on startup with `ADMIN_PASSWORD must be set...` | `ADMIN_PASSWORD` is empty, shorter than 8 characters or `admin`. |
| `docker compose` fails with `required variable ... is missing a value` | `RTSP_USER`, `RTSP_PASSWORD` or `SITE_ADDRESS` is not set in `.env`. |
| `Bind for 0.0.0.0:3000 failed: port is already allocated` (development) | Another service uses port 3000 on this machine; stop it or free the port. |

---

## Configuration reference

All configuration is read from `.env` (see `.env.example` for the full list of keys).

**Security and access**

| Variable | Purpose |
|---|---|
| `ADMIN_PASSWORD` | Password of the `admin` user (required, at least 8 characters, not `admin`) |
| `SECRET_KEY` / `ALGORITHM` | JWT signing configuration — set `SECRET_KEY` to a long random value |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | JWT access token lifetime |
| `SITE_ADDRESS` | Addresses Caddy serves over HTTPS, e.g. `https://192.168.1.14, https://raspberrypi.local` (required) |
| `DEFAULT_SNI` | Bare IP from `SITE_ADDRESS` whose certificate is used when a client connects by IP (clients send no SNI for IPs) |
| `CORS_ORIGINS` | Comma-separated origins allowed to call the API cross-origin; empty in production, the frontend origin in development |
| `MANAGER_UPSTREAM` | Where Caddy finds the frontend (default `host.docker.internal:5173`) |

**Camera and recording**

| Variable | Purpose |
|---|---|
| `RTSP_USER` / `RTSP_PASSWORD` | Credentials of the go2rtc RTSP server and web UI (required) |
| `RTSP_PORT` | go2rtc RTSP port (default `8554`) |
| `LIVE_CAMERA` | go2rtc stream served by the API's live view (default `cam1`) |
| `RECORD_CAMERAS` | Space-separated go2rtc streams to record (default `cam1`); names must not contain `_` |
| `RECORD_SEGMENT_SECONDS` | Length of each recorded segment (default `3600`) |

**Internal services**

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

The VLM server (`qwen-api`) is configured in `docker-compose.yml` and `llama.cpp/scripts/`: `start_qwen.sh` holds the `llama.cpp` flags and `download_models.sh` fetches the GGUF weights.

---

## Authentication

- The `admin` user (`is_admin=true`) takes its password from `ADMIN_PASSWORD` on every API boot: it is created if missing and updated if the password changed. Tokens issued before a password change stay valid until they expire (`ACCESS_TOKEN_EXPIRE_MINUTES`).
- `POST /login` exchanges credentials for a JWT bearer token.
- `POST /register` lets an existing admin create new (non-admin) users.
- Protected routes accept the token either as a standard `Authorization: Bearer <token>` header or, for endpoints consumed directly by `<video>`/`<img>` tags, as a `?token=` query parameter.

---

## API overview

All routes below (except `/healthcheck`, `/login`, `/register`) require a valid bearer token. In production they are served under `https://<pi-address>/api` (e.g. `/api/login`); in development directly on `http://localhost:3000`.

| Method | Route | Description |
|---|---|---|
| `GET` | `/healthcheck` | Liveness check |
| `POST` | `/login` | Authenticate and receive a JWT |
| `POST` | `/register` | Create a new user (admin only) |
| `GET` | `/videos/` | Paginated list of recorded segments, optionally filtered by date range |
| `GET` | `/videos/live` | MJPEG live stream of `LIVE_CAMERA`, read from go2rtc |
| `GET` | `/videos/{video_name}/playback` | Range-request video streaming for playback |
| `GET` | `/videos/{video_name}/thumbnail` | JPEG thumbnail for a segment |
| `POST` | `/query` | Semantic search over captioned events (natural-language text query) |
| `POST` | `/query/image` | Reverse image search (uploaded image or a frame from a recording) |
| `GET` | `/metrics/daily` | Event/object analytics for a date or date range (`start_date`/`end_date`, default: today) |
| `GET` | `/metrics/overall` | All-time event/object analytics |

Interactive OpenAPI docs: `https://<pi-address>/api/docs` (production) or `http://localhost:3000/docs` (development).

---

## Development notes

- The three `guard/*` services (`pi-guard`, `guard-worker`, `guard-enrichment-worker`) share the `pi-guard:latest` image, built once by `pi-guard`, and differ only by the command they run.
- `guard/infrastructure/di/container.py` and `guard/worker/*container.py` wire concrete adapters (ChromaDB, SQLite, Redis, CLIP, VLMs) into the domain services; swap an adapter by changing what's constructed there, not by touching `core`.
- SQLite runs in WAL mode so the API (reads) and the workers (writes) can safely share the same database file across containers.
- In development, `guard/` is bind-mounted into the three `guard/*` containers and the API runs with `--reload`, so code changes are picked up without rebuilding (the workers still need a restart). In production the code is baked into the image.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
