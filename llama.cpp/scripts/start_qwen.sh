#!/bin/sh

/llama-server \
  -m /models/Qwen3.5-0.8B-Q8_0.gguf \
  --mmproj /models/Qwen3.5-0.8B-mmproj-F16.gguf \
  -c 4096 \
  -t 3 \
  -n 1024 \
  --jinja \
  --image-max-tokens 256 \
  --reasoning off \
  --reasoning-budget 0 \
  --port 8081 \
  --host 0.0.0.0