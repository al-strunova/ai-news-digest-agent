"""Shared types and helpers used by every fetcher.

Each fetcher returns a FetchResult containing zero or more Items.
strip_html lives here because both the RSS fetcher and the HN fetcher
need plain-text snippet conversion with the same shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from bs4 import BeautifulSoup


@dataclass
class Item:
    title: str
    source_name: str
    url: str
    published: str                      # ISO 8601 if the feed gave us a parseable date, else raw string
    content_snippet: str                # plain text, capped at SNIPPET_CHARS by the caller
    linked_url: Optional[str] = None    # populated only by HN: the external article the story points to


@dataclass
class FetchResult:
    source_name: str
    success: bool
    items: list[Item] = field(default_factory=list)
    error: Optional[str] = None


def strip_html(html: str) -> str:
    """Strip HTML tags, collapse runs of whitespace into a single space.

    Suitable for inline snippets. Producing identical output across all
    fetchers means the curator sees consistently shaped content.
    """
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()