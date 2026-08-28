#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage: docker/build-agent-images.sh [base|codex|openclaw|hermes|claudecode|workbuddy|all]...

With no arguments, builds the public Quickstart images: base and Codex.
Every runtime is built from pinned packages or source revisions available from
its public upstream; no local vendor archive is required.
EOF
}

if [[ "${1:-}" == "--list" ]]; then
  printf '%s\n' base codex openclaw hermes claudecode workbuddy
  exit 0
fi
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -eq 0 ]]; then
  set -- codex
fi
if [[ " $* " == *" all "* ]]; then
  set -- codex openclaw hermes claudecode workbuddy
fi

base_image="${SANDBOX_BASE_IMAGE:-ubuntu:24.04}"
npm_registry="${NPM_REGISTRY:-https://registry.npmjs.org}"
docker build --build-arg "BASE_IMAGE=$base_image" -f docker/red-sandbox-base/Dockerfile -t red-sandbox-base:v1 .

for runtime in "$@"; do
  case "$runtime" in
    base)
      ;;
    codex)
      docker build --build-arg "NPM_REGISTRY=$npm_registry" -f docker/red-agent-codex/Dockerfile -t red-agent-world-codex:v1 .
      ;;
    openclaw)
      docker build -f docker/red-agent-openclaw/Dockerfile -t red-agent-world-openclaw:v1 .
      ;;
    claude|claudecode)
      docker build -f docker/red-agent-claudecode/Dockerfile -t red-agent-world-claudecode:v1 .
      ;;
    hermes)
      docker build -f docker/red-agent-hermes/Dockerfile -t red-agent-world-hermes:v1 .
      ;;
    workbuddy)
      docker build -f docker/red-agent-workbuddy/Dockerfile -t red-agent-world-workbuddy:v1 .
      ;;
    *)
      echo "Unknown runtime: $runtime" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "Built runtime images: $*"
