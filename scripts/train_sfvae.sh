#!/usr/bin/env bash
set -euo pipefail
python -m geoembed_twin sfvae-train --fast "$@"
