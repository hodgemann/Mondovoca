#!/usr/bin/env bash
set -euo pipefail
MONDOVOCA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MONDOVOCA_PYTHON="${MONDOVOCA_PYTHON:-$MONDOVOCA_ROOT/.venv/bin/python}"
if [[ ! -x "$MONDOVOCA_PYTHON" ]]; then MONDOVOCA_PYTHON="$(command -v python3)"; fi
exec "$MONDOVOCA_PYTHON" "$MONDOVOCA_ROOT/scripts/mondovoca_service.py" start
