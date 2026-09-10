#!/bin/sh
exec /app/llama-server \
  -m /models/plano-orchestrator-4b.gguf \
  --alias plano-orchestrator-4b \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 8192 \
  --threads "${LLAMA_THREADS:-2}" \
  --api-key "${LLAMA_API_KEY}"