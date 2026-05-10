"""State persistence helpers.

Three files under data/:
  seen_items.json       URL -> ISO timestamp of items already covered.
                        Pruned to last 14 days each run.
  source_health.json    per-source list of recent run records (success
                        + error). Last 14 entries kept per source.
  token_usage.jsonl     append-only, one JSON object per run.

We deliberately don't catch JSONDecodeError on load: corrupted state is
a bug worth surfacing as a failed run rather than silently resetting.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from config import HEALTH_RUNS_KEPT, SEEN_TTL_DAYS

DATA_DIR = Path(__file__).parent / "data"
SEEN_ITEMS_PATH = DATA_DIR / "seen_items.json"
SOURCE_HEALTH_PATH = DATA_DIR / "source_health.json"
TOKEN_USAGE_PATH = DATA_DIR / "token_usage.jsonl"


def _atomic_write_text(path: Path, text: str) -> None:
    """Write to a sibling .tmp file and atomically rename into place.

    On POSIX, Path.replace is atomic — the destination either has the
    old contents or the new, never partial. Protects against corruption
    if the runner is preempted mid-write.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def load_seen_items() -> dict[str, str]:
    if not SEEN_ITEMS_PATH.exists():
        return {}
    data = json.loads(SEEN_ITEMS_PATH.read_text())
    return data.get("items", {})


def prune_expired(seen: dict[str, str]) -> dict[str, str]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=SEEN_TTL_DAYS)
    kept: dict[str, str] = {}
    for url, ts in seen.items():
        try:
            t = datetime.fromisoformat(ts)
        except ValueError:
            continue
        if t >= cutoff:
            kept[url] = ts
    return kept


def save_seen_items(seen: dict[str, str]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        SEEN_ITEMS_PATH,
        json.dumps({"items": seen}, indent=2, sort_keys=True),
    )


def load_source_health() -> dict:
    if not SOURCE_HEALTH_PATH.exists():
        return {"sources": {}}
    return json.loads(SOURCE_HEALTH_PATH.read_text())


def record_source_run(
    health: dict,
    source_name: str,
    success: bool,
    error: str | None,
) -> None:
    sources = health.setdefault("sources", {})
    record = sources.setdefault(source_name, {"recent_runs": []})
    record["recent_runs"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "error": error,
    })
    record["recent_runs"] = record["recent_runs"][-HEALTH_RUNS_KEPT:]


def save_source_health(health: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        SOURCE_HEALTH_PATH,
        json.dumps(health, indent=2, sort_keys=True),
    )


def append_token_usage(row: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TOKEN_USAGE_PATH.open("a") as f:
        f.write(json.dumps(row) + "\n")