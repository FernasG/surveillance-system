#!/bin/sh

/llama-server \
  -m /models/gemma-4-E2B-it-qat.Q4_K_M.gguf \
  --mmproj /models/gemma-4-E2B-it-qat-mmproj-F16.gguf \
  -c 4096 \
  -t 3 \
  -n 1024 \
  --jinja \
  --image-max-tokens 256 \
  --reasoning off \
  --reasoning-budget 0 \
  --port 8082 \
  --host 0.0.0.0