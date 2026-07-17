#!/bin/sh

/llama-server \
  -m /models/LFM2.5-VL-1.6B-Q5_K_S.gguf \
  --mmproj /models/LFM2.5-VL-1.6B-mmproj-F16.gguf \
  -c 4096 \
  -t 3 \
  -n 1024 \
  --jinja \
  --image-max-tokens 256 \
  --reasoning off \
  --reasoning-budget 0 \
  --port 8081 \
  --host 0.0.0.0