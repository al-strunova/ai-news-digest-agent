"""Anthropic news index scraper.

The Anthropic news index doesn't expose RSS, so we scrape the listing
page for title, date, and URL of each post. Snippet is left empty;
the curator can call fetch_full_article to read individual posts.

Selectors verified against https://www.anthropic.com/news on 2026-05-02.
The page is server-rendered so plain requests + BeautifulSoup is enough,
no headless browser required.
"""

from __future__ import annotations

from datetime import datetime, timezone
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import ANTHROPIC_BASE_URL, HTTP_TIMEOUT_SECONDS, USER_AGENT
from fetchers.common import FetchResult, Item
from sources import Source


def _parse_date(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    try:
        return datetime.strptime(text, "%b %d, %Y").replace(tzinfo=timezone.utc).isoformat()
    except ValueError:
        return text


def fetch(source: Source) -> FetchResult:
    if source.kind != "anthropic_blog":
        return FetchResult(source.name, False, error=f"wrong kind: {source.kind}")

    try:
        resp = requests.get(
            source.url,
            timeout=HTTP_TIMEOUT_SECONDS,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return FetchResult(source.name, False, error=f"fetch failed: {e}")

    soup = BeautifulSoup(resp.text, "html.parser")
    anchors = soup.select('a[href^="/news/"]')

    seen: set[str] = set()
    items: list[Item] = []
    for a in anchors:
        href = a.get("href", "")
        if href == "/news" or href in seen:
            continue
        seen.add(href)

        title_el = a.select_one('[class*="title"], h1, h2, h3, h4')
        title = (title_el.get_text(strip=True) if title_el
                 else a.get_text(separator=" ", strip=True)).strip()
        if not title:
            continue

        time_el = a.find("time")
        published = _parse_date(time_el.get_text(strip=True)) if time_el else ""

        items.append(Item(
            title=title,
            source_name=source.name,
            url=urljoin(ANTHROPIC_BASE_URL, href),
            published=published,
            content_snippet="",
        ))

    return FetchResult(source.name, True, items)