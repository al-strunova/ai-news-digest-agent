"""Generic RSS fetcher.

One source in, one FetchResult out. Errors are returned, not raised, so
main.py can keep going when a single feed breaks.

Requests are sent with feedparser's bytes via `requests` rather than
letting feedparser fetch the URL itself, so we can attach the full set of
headers a normal HTTP client would send (Accept, Accept-Language,
Accept-Encoding). Some CDN-fronted feeds (Substack/Cloudflare) 403 on
header sets that look stripped-down, even when the User-Agent is fine.
"""

from __future__ import annotations

from datetime import datetime, timezone

import feedparser
import requests

from config import HTTP_TIMEOUT_SECONDS, SNIPPET_CHARS, USER_AGENT
from fetchers.common import FetchResult, Item, strip_html
from sources import Source

# Advertise only encodings `requests` can decode without extra deps
# (gzip, deflate). Brotli would need the `brotli` package.
_REQUEST_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "application/rss+xml, application/atom+xml, "
        "application/xml;q=0.9, text/xml;q=0.8, */*;q=0.5"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
}


def _parse_published(entry) -> str:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        return datetime(*parsed[:6], tzinfo=timezone.utc).isoformat()
    return entry.get("published") or entry.get("updated") or ""


def _entry_content(entry) -> str:
    if "content" in entry and entry.content:
        return entry.content[0].get("value", "")
    return entry.get("summary") or entry.get("description") or ""


def fetch(source: Source) -> FetchResult:
    if source.kind != "rss":
        return FetchResult(source.name, False, error=f"not an RSS source: kind={source.kind}")

    try:
        resp = requests.get(
            source.url,
            headers=_REQUEST_HEADERS,
            timeout=HTTP_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
    except requests.RequestException as e:
        return FetchResult(source.name, False, error=f"fetch failed: {e}")

    if resp.status_code >= 400:
        return FetchResult(source.name, False, error=f"HTTP {resp.status_code}")

    parsed = feedparser.parse(resp.content)

    if parsed.bozo and not parsed.entries:
        return FetchResult(source.name, False, error=f"feed parse error: {parsed.bozo_exception}")

    items: list[Item] = []
    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        url = (entry.get("link") or "").strip()
        if not title or not url:
            continue
        snippet = strip_html(_entry_content(entry))[:SNIPPET_CHARS]
        items.append(Item(
            title=title,
            source_name=source.name,
            url=url,
            published=_parse_published(entry),
            content_snippet=snippet,
        ))

    return FetchResult(source.name, True, items)