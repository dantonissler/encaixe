#!/usr/bin/env bash
# Roda o promptfoo na versão fixada, sem telemetria nem checagem de atualização.
# O promptfoo 0.125.1 exige Node >= 22.22; se o node do PATH for mais antigo, tenta o do nvm.
set -euo pipefail

PROMPTFOO_VERSION="0.125.1"
export PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE=1

node_ok() {
  command -v node >/dev/null 2>&1 || return 1
  node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=22)?0:1)'
}

if ! node_ok; then
  for dir in $(ls -d "${NVM_DIR:-$HOME/.nvm}"/versions/node/v2[2-9]* 2>/dev/null | sort -V -r); do
    PATH="$dir/bin:$PATH"
    node_ok && break
  done
fi
node_ok || { echo "promptfoo ${PROMPTFOO_VERSION} exige Node >= 22.22 (instale com nvm)." >&2; exit 2; }

exec npx -y "promptfoo@${PROMPTFOO_VERSION}" "$@"
