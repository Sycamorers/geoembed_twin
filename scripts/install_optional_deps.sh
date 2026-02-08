#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

missing="$($PYTHON_BIN - <<'PY'
import importlib.util
from pathlib import Path

req_map = {
    "geomloss": "geomloss>=0.2.6",
    "faiss": "faiss-cpu>=1.7.4",
    "open3d": "open3d>=0.17",
    "rich": "rich>=13.7",
    "pytest": "pytest>=7.4",
}

missing = []
for module, requirement in req_map.items():
    if importlib.util.find_spec(module) is None:
        missing.append(requirement)

print(" ".join(missing))
PY
)"

if [[ -z "${missing// }" ]]; then
  echo "All optional dependencies already present."
  exit 0
fi

echo "Installing missing optional packages: ${missing}"
"${PYTHON_BIN}" -m pip install --no-input --no-color ${missing}
echo "Optional dependencies installed (safe to re-run)."
