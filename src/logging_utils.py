"""
logging_utils.py — Structured request logging to a JSONL file.

Every question answered by the agent is appended as a single JSON line to
logs/requests.jsonl. This gives me a permanent audit trail of what was asked,
how it was routed, which provider answered, and whether it was refused.

Using JSONL (one JSON object per line) instead of a database keeps the logging
zero-dependency and makes it trivial to load into pandas for analysis:

    import pandas as pd
    df = pd.read_json("logs/requests.jsonl", lines=True)

PII note: only the masked question is logged (PII replaced with [EMAIL] etc.),
never the raw user input. The session ID is also hashed so it can't be traced
back to a specific user.
"""

import hashlib
import json
from datetime import datetime, timezone

from src.config import settings


def hash_id(value: str) -> str:
    """
    Return the first 8 hex characters of a SHA-256 hash.

    Used to anonymise session/thread IDs in logs — short enough to be
    readable in the log file, long enough to distinguish sessions.
    """
    return hashlib.sha256(value.encode()).hexdigest()[:8]


def log_request(record: dict) -> None:
    """
    Append one structured record to logs/requests.jsonl.

    Silently swallows exceptions so a logging failure never crashes a user
    request. The timestamp is always in UTC ISO-8601 format.

    Args:
        record: Dict of fields to log. A "ts" (timestamp) field is added automatically.
    """
    try:
        settings.log_dir.mkdir(exist_ok=True)
        line = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **record,
        }
        with open(settings.log_dir / "requests.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    except Exception:
        pass  # logging should never break a user-facing request
