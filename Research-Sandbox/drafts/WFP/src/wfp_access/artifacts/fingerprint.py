from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_fingerprint(payload: Any) -> str:
    """Create a stable short fingerprint from JSON-compatible metadata."""

    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]
