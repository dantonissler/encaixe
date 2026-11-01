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

# Modelos Hugging Face: baixados uma única vez aqui; em runtime o projeto roda
# com HF_HUB_OFFLINE=1 (ver Makefile). Prompt-Guard (gated) NÃO é baixado.
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('BAAI/bge-reranker-v2-m3')
snapshot_download('BAAI/bge-m3', allow_patterns=[
    'tokenizer*', 'special_tokens_map.json', 'sentencepiece*', '*.model', 'config*.json'])
"

# Modelos do Docling (layout, tableformer, code/formula, classificador, OCR) no cache padrão
uv run docling-tools models download
