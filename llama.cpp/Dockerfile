FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y build-essential git cmake

RUN git clone https://github.com/ggerganov/llama.cpp.git /app
WORKDIR /app

RUN cmake -B build \
    && cmake --build build --config Release -j 4 --target llama-server

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/build/bin /app/build/bin
COPY --from=builder /app/build/src /app/build/src

ENV LD_LIBRARY_PATH=/app/build/bin:/app/build/src

RUN ln -s /app/build/bin/llama-server /llama-server