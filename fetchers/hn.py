"""Hacker News fetcher via the Algolia API.

Pulls recent stories (last 48 hours) with points >= 50, then keeps only
ones whose title matches an AI/ML keyword pattern. Threshold and window
are intentionally generous; tune after a week of real data.

The Algolia search_by_date endpoint is unauthenticated and rate-limited
to ~10,000 requests/hour per IP. Fine for a once-a-day cadence.

Each Item carries a linked_url pointing at the external article the HN
story is about. The curator's fetch_full_article tool extends its
allowlist with these linked_urls each run, so the model can fetch the
actual articles, not just the HN discussion pages.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

import requests

from config import (
    HN_HITS_PER_PAGE,
    HN_LOOKBACK_HOURS,
    HN_POINTS_FLOOR,
    HTTP_TIMEOUT_SECONDS,
    SNIPPET_CHARS,
    USER_AGENT,
)
from fetchers.common import FetchResult, Item, strip_html
from sources import Source

# Word-boundary patterns. Bare "agent" / "prompt" / "model" excluded:
# too overloaded with non-AI senses to be useful filters.
KEYWORD_PATTERNS = [
    r"\bAI\b", r"\bAGI\b", r"\bML\b",
    r"\bLLM\b", r"\bGPT\b", r"\bRAG\b", r"\bRLHF\b", r"\bMCP\b",
    r"\bClaude\b", r"\bGemini\b", r"\bChatGPT\b",
    r"\bLlama\b", r"\bMistral\b", r"\bDeepSeek\b", r"\bGrok\b",
    r"\bSora\b", r"\bDALL-E\b", r"\bMidjourney\b",
    r"\bAnthropic\b", r"\bOpenAI\b", r"\bDeepMind\b",
    r"\bHugging\s*Face\b",
    r"\bagentic\b", r"\bAI\s+agent\b",
    r"\btransformer\b", r"\bdiffusion\b", r"\bneural\b",
    r"\bembedding\b", r"\bvLLM\b",
    r"\bmachine\s+learning\b", r"\bdeep\s+learning\b",
    r"\breinforcement\s+learning\b",
    r"\bfine[-\s]?tuning\b", r"\binference\b",
    r"\bcontext\s+window\b",
]
KEYWORD_RE = re.compile("|".join(KEYWORD_PATTERNS), re.IGNORECASE)


def fetch(source: Source) -> FetchResult:
    if source.kind != "hn":
        return FetchResult(source.name, False, error=f"wrong kind: {source.kind}")

    cutoff = int(time.time()) - HN_LOOKBACK_HOURS * 3600
    params = {
        "tags": "story",
        "numericFilters": f"points>={HN_POINTS_FLOOR},created_at_i>{cutoff}",
        "hitsPerPage": HN_HITS_PER_PAGE,
    }

    try:
        resp = requests.get(
            source.url,
            params=params,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        return FetchResult(source.name, False, error=f"fetch failed: {e}")
    except ValueError as e:
        return FetchResult(source.name, False, error=f"json parse failed: {e}")

    items: list[Item] = []
    for h in data.get("hits", []):
        title = (h.get("title") or "").strip()
        object_id = h.get("objectID")
        if not title or not object_id:
            continue
        if not KEYWORD_RE.search(title):
            continue

        article_url = (h.get("url") or "").strip() or None
        author = h.get("author") or "?"
        points = h.get("points") or 0
        comments = h.get("num_comments") or 0
        story_text = strip_html(h.get("story_text") or "")

        snippet_parts = [f"{points} points | {comments} comments | by {author}"]
        if article_url:
            snippet_parts.append(f"Linked: {article_url}")
        if story_text:
            snippet_parts.append(story_text)
        snippet = "\n".join(snippet_parts)[:SNIPPET_CHARS]

        created = h.get("created_at_i")
        if created:
            published = datetime.fromtimestamp(created, tz=timezone.utc).isoformat()
        else:
            published = h.get("created_at") or ""

        items.append(Item(
            title=title,
            source_name=source.name,
            url=f"https://news.ycombinator.com/item?id={object_id}",
            published=published,
            content_snippet=snippet,
            linked_url=article_url,
        ))

    return FetchResult(source.name, True, items)