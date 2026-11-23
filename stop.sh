#!/usr/bin/env bash
set -euo pipefail
MONDOVOCA_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$MONDOVOCA_ROOT/scripts/mondovoca_service.py" stop
