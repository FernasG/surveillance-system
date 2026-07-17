#!/bin/sh

apk add --no-cache wget

echo "Checking Liquid LFM models..."
[ -f /models/LFM2.5-VL-1.6B-Q5_K_S.gguf ] || wget -O /models/LFM2.5-VL-1.6B-Q5_K_S.gguf https://huggingface.co/unsloth/LFM2.5-VL-1.6B-GGUF/resolve/main/LFM2.5-VL-1.6B-Q5_K_S.gguf
[ -f /models/LFM2.5-VL-1.6B-mmproj-F16.gguf ] || wget -O /models/LFM2.5-VL-1.6B-mmproj-F16.gguf https://huggingface.co/unsloth/LFM2.5-VL-1.6B-GGUF/resolve/main/mmproj-F16.gguf

echo "Checking Qwen models..."
[ -f /models/Qwen3.5-0.8B-Q8_0.gguf ] || wget -O /models/Qwen3.5-0.8B-Q8_0.gguf https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf
[ -f /models/Qwen3.5-0.8B-mmproj-F16.gguf ] || wget -O /models/Qwen3.5-0.8B-mmproj-F16.gguf https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/mmproj-F16.gguf

echo "Checking Gemma models..."
[ -f /models/gemma-4-E2B-it-qat.Q4_K_M.gguf ] || wget -O /models/gemma-4-E2B-it-qat.Q4_K_M.gguf https://huggingface.co/prithivMLmods/gemma-4-E2B-it-qat-GGUF/resolve/main/gemma-4-E2B-it-qat.Q4_K_M.gguf
[ -f /models/gemma-4-E2B-it-qat-mmproj-F16.gguf ] || wget -O /models/gemma-4-E2B-it-qat-mmproj-F16.gguf https://huggingface.co/prithivMLmods/gemma-4-E2B-it-qat-GGUF/resolve/main/gemma-4-E2B-it-qat.mmproj-f16.gguf

echo "All models are ready!"