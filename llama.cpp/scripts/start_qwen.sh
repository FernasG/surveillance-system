#!/bin/sh

/llama-server \
  -m /models/mtp-Qwen3.5-0.8B-UD-Q8_K_XL.gguf \
  --mmproj /models/Qwen3.5-0.8B-mmproj-F16.gguf \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 --spec-draft-n-min 0 \
  -c 4096 \
  -t 3 \
  --parallel 1 \
  --cpu-strict-batch 1 \
  --flash-attn on --kv-unified \
  --jinja \
  --image-min-tokens 1024 \
  --reasoning off \
  --reasoning-budget 0 \
  --port 8081 \
  --host 0.0.0.0