#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR" python -m geoembed_twin sfvae-train --fast "$@"
