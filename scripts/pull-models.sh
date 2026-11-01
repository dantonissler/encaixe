#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OLLAMA="docker compose exec -T ollama ollama"

for m in qwen3.5:4b llama3.2:3b llama-guard3:1b bge-m3; do
  $OLLAMA pull "$m"
done

for alias in llama3.2-3b-cpu llama-guard3-1b-cpu bge-m3-cpu; do
  docker compose cp "docker/ollama/Modelfile.$alias" "ollama:/tmp/Modelfile.$alias"
  $OLLAMA create "$alias" -f "/tmp/Modelfile.$alias"
done
