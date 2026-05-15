"""Daily AI/ML news digest agent — entry point.

Run order:
  1. Load env, prune state files
  2. Fetch from every source (failures isolated, run continues)
  3. Filter items: time window, then dedup against seen_items
  4. Curate via Claude (timed)
  5. Build digest body, append source-health footer
  6. Send email (always — quiet days look the same as broken agent days)
  7. Persist state and observability:
       - source_health.json   always
       - token_usage.jsonl    always
       - seen_items.json      only if email sent AND curate succeeded
                              (a failed send must not lose items, and
                              an error email must not mark them seen)
  8. Exit 1 on curate failure or email failure, else 0
"""

from __future__ import annotations

import re
import sys
import time
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv

from config import ITEM_AGE_LIMIT_DAYS, MODEL
from curator import CurateResult, curate
from emailer import send_digest
from fetchers import anthropic_blog, hn, rss
from fetchers.common import FetchResult, Item
from sources import SOURCES, Source
from state import (
    append_token_usage,
    load_seen_items,
    load_source_health,
    prune_expired,
    record_source_run,
    save_seen_items,
    save_source_health,
)


# Provider exceptions routinely embed account identifiers (org UUIDs,
# request IDs) and the recipient address back from the API. Strip those
# before the message hits token_usage.jsonl or the digest email body —
# the repo is public, so anything written to disk is published.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.\w+")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_API_KEY_RE = re.compile(r"\b(?:sk-[A-Za-z0-9_-]+|AIza[A-Za-z0-9_-]+|re_[A-Za-z0-9_-]+)\b")
_REQ_ID_RE = re.compile(r"\breq_[A-Za-z0-9]+\b")


def sanitize_error(e: BaseException) -> str:
    msg = str(e)
    msg = _EMAIL_RE.sub("[email]", msg)
    msg = _API_KEY_RE.sub("[apikey]", msg)
    msg = _UUID_RE.sub("[uuid]", msg)
    msg = _REQ_ID_RE.sub("[reqid]", msg)
    return f"{type(e).__name__}: {msg}"


def fetch_one(source: Source) -> FetchResult:
    if source.kind == "rss":
        return rss.fetch(source)
    if source.kind == "anthropic_blog":
        return anthropic_blog.fetch(source)
    if source.kind == "hn":
        return hn.fetch(source)
    return FetchResult(source.name, False, error=f"unknown source kind: {source.kind}")


def fetch_all(sources: list[Source]) -> list[FetchResult]:
    return [fetch_one(s) for s in sources]


def collect_items(results: list[FetchResult]) -> list[Item]:
    out: list[Item] = []
    for r in results:
        if r.success:
            out.extend(r.items)
    return out


def filter_recent(items: list[Item], days: int) -> list[Item]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    out: list[Item] = []
    for it in items:
        if not it.published:
            print(f"warn: missing date, keeping item: {it.source_name} | {it.url}", file=sys.stderr)
            out.append(it)
            continue
        try:
            t = datetime.fromisoformat(it.published)
        except ValueError:
            print(f"warn: unparseable date {it.published!r}, keeping item: {it.source_name} | {it.url}", file=sys.stderr)
            out.append(it)
            continue
        if t >= cutoff:
            out.append(it)
    return out


def filter_unseen(items: list[Item], seen: dict[str, str]) -> list[Item]:
    return [it for it in items if it.url not in seen]


def build_health_footer(results: list[FetchResult]) -> str:
    failures = [r for r in results if not r.success]
    if not failures:
        return ""
    lines = ["", "---", "", "*Source health:*", ""]
    for r in failures:
        lines.append(f"- {r.source_name}: {r.error}")
    return "\n".join(lines)


def empty_curate_result() -> CurateResult:
    return CurateResult(
        digest_markdown="",
        tool_calls=0,
        items_input=0,
        items_output=0,
        input_tokens=0,
        output_tokens=0,
        cache_read_tokens=0,
        cache_creation_tokens=0,
    )


def main() -> int:
    load_dotenv()
    run_started = time.monotonic()
    today_date = datetime.now(timezone.utc).date().isoformat()

    seen = prune_expired(load_seen_items())
    health = load_source_health()

    results = fetch_all(SOURCES)
    raw = collect_items(results)
    fresh = filter_recent(raw, ITEM_AGE_LIMIT_DAYS)
    unseen = filter_unseen(fresh, seen)

    curate_started = time.monotonic()
    curate_error: str | None = None
    if unseen:
        try:
            result = curate(unseen, SOURCES, today=today_date)
        except Exception as e:
            curate_error = sanitize_error(e)
            result = empty_curate_result()
    else:
        result = empty_curate_result()
    curate_duration = time.monotonic() - curate_started

    if curate_error:
        digest_md = (
            f"Agent error during curation:\n\n"
            f"```\n{curate_error}\n```\n\n"
            f"No digest produced this run."
        )
    elif result.digest_markdown.strip():
        digest_md = result.digest_markdown.strip()
    else:
        digest_md = "No notable items today."
    digest_md += build_health_footer(results)

    subject = f"AI/ML digest — {today_date}"
    email_error: str | None = None
    message_id: str | None = None
    try:
        message_id = send_digest(digest_md, subject)
    except Exception as e:
        email_error = sanitize_error(e)
        print(f"email send failed: {email_error}", file=sys.stderr)

    # Always: record source health
    for r in results:
        record_source_run(health, r.source_name, r.success, r.error)
    save_source_health(health)

    # Always: log token usage
    append_token_usage({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "cache_read_tokens": result.cache_read_tokens,
        "cache_creation_tokens": result.cache_creation_tokens,
        "tool_calls": result.tool_calls,
        "items_input": result.items_input,
        "items_output": result.items_output,
        "sources_succeeded": sum(1 for r in results if r.success),
        "sources_failed": sum(1 for r in results if not r.success),
        "duration_seconds": round(curate_duration, 2),
        "curate_error": curate_error,
        "email_error": email_error,
    })

    # Mark items seen only when the digest email landed AND curation
    # actually ran. An error email still sets message_id, but its items
    # were never curated — leaving curate_error out here would silently
    # burn them so the next (recovered) run never sees them again.
    if message_id is not None and curate_error is None:
        now_iso = datetime.now(timezone.utc).isoformat()
        for it in unseen:
            seen[it.url] = now_iso
        save_seen_items(seen)

    succ = sum(1 for r in results if r.success)
    total = len(results)
    run_duration = time.monotonic() - run_started
    print(
        f"digest: {result.items_output} items from {succ}/{total} sources, "
        f"{result.input_tokens} in / {result.output_tokens} out, "
        f"{result.tool_calls} tool calls, {curate_duration:.1f}s curate, "
        f"{run_duration:.1f}s total"
        + (f" | curate_error={curate_error}" if curate_error else "")
        + (f" | email_error={email_error}" if email_error else "")
    )

    if curate_error or email_error:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())