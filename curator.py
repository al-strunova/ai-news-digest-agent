"""Curation step.

Takes the list of items collected by the fetchers, hands them to an LLM
with one tool (fetch_full_article), and returns the markdown digest plus
token usage data for the per-run JSONL log.

Provider-agnostic: the LLM tool-use loop is implemented per-provider in
providers/anthropic.py and providers/gemini.py, both exposing the same
run_tool_use_loop signature. config.LLM_PROVIDER picks one. Adding a
third provider is one new providers/<name>.py plus one elif below.

The tool's allowlist is built per run from two sources:
  - static_hosts:  hostnames extracted from each source's homepage
                   (HN's homepage covers news.ycombinator.com)
  - runtime_hosts: hostnames extracted from each HN item's linked_url
                   for the current run

Both lists are exact-host match (no subdomain fuzzing). This keeps cute
attacks like "evil.openai.com" out and avoids the substack.com gotcha
where any *.substack.com would otherwise be reachable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

import config
from config import (
    HTTP_TIMEOUT_SECONDS,
    MAX_BYTES,
    MAX_OUTPUT_TOKENS,
    MAX_REDIRECTS,
    MAX_RETURN_CHARS,
    TOOL_CALL_CAP,
    USER_AGENT,
)
from fetchers.common import Item
from sources import Source


SYSTEM_PROMPT = """\
You are a curator producing a daily AI/ML news digest for a working AI
engineer. Your output is the digest itself, in markdown. Do not include
any preamble, meta-commentary, or notes about your process.

# Audience

The reader works in AI engineering and wants to stay broadly informed
across AI engineering, ML research, and where the field is moving as a
whole. They are not narrowed to any one subdomain. The point is breadth:
surface what is notable across the field, including areas they do not
already work on, so they catch shifts they would otherwise miss.

# What to surface

Be inclusive, not selective. Filter out clear junk. Everything else,
surface it. The reader prefers a longer digest with some borderline
items over a tight digest that drops real signal. Do not gatekeep.

Items worth surfacing include:
- Major lab announcements: model releases, capability milestones,
  infrastructure shifts.
- Research being discussed by practitioners: papers, evaluations,
  methodology, benchmarks.
- Tools and frameworks with real adoption signal. Not "we built X" but
  "X is being used because Y".
- Shifts in field consensus: when conventional wisdom changes, when an
  approach gains or loses traction.
- Learning resources: substantive courses, talks, books, deep-dive blog
  posts.
- Partnerships and business news that change something material. Weight
  by substance, not category. Cloud and compute deals signal stack
  direction; enterprise wins signal market share; government MOUs
  signal regulation; large funding rounds signal where money is
  flowing. Keep these. Cut pure PR with no material change.

# What to cut

- Hype listicles ("Top 10 AI tools for X").
- Pure vendor marketing dressed as news.
- Re-announcements of items already covered.
- Tabloid framing about AI sentience, celebrity AI takes, opinion
  pieces with no substance.
- Generic "Company X adds AI to product Y" press releases unless the
  integration is technically interesting.

When uncertain, include. The human can scan past borderline items;
missing real signal is the worse failure.

# Volume

Match the day. Do not pad to a target. Do not cut to hit a number.
- Quiet day: 5 to 8 items
- Average day: 8 to 15 items
- Active news day: 15 to 25 items
- Major news day (model release, big paper, etc.): 25 or more

# Format

Output markdown. Group items into sections only when a section has more
than one item. Skip empty sections.

Possible sections (use whatever fits the day's items):
- Major Announcements
- Research
- Tools and Frameworks
- Discussion (community-driven items: HN threads with substantive
  technical discussion, comparison threads, ongoing debates worth
  following)

For each item:
- A short title as an H3 heading that links to the item URL
- 2 to 3 sentences explaining why it matters, written as a plain
  paragraph immediately below the title. Focus on substance: what
  changed, why a working AI engineer would want to know. Avoid
  restating the title. The summary is body text — do not prefix it
  with `#`, `##`, or any other heading marker.

Example:
### [Anthropic releases Claude Opus 4.7](https://www.anthropic.com/news/claude-opus-4-7)
Stronger performance across coding, agents, vision, and multi-step
tasks. Notable for engineers because the agentic-coding gains affect
tool-use reliability in production loops, where prior Opus releases
were already strong.

# Tools

You have one tool: fetch_full_article(url). Use it when:
- The provided snippet is intro fluff and you cannot tell whether the
  item matters.
- You need more context to write a substantive 2 to 3 sentence summary.
- The item is from Hacker News and you want to read the linked article
  rather than just the discussion page.

Be selective. You have a hard cap of 20 tool calls per run. Skim before
fetching. Do not fetch items you would cut anyway.

For Hacker News items: each HN item has a Linked: field showing the
article URL the post is about. The HN page itself usually just has
comments. To judge or summarize an HN-surfaced story, fetch the linked
article URL (allowed by the runtime allowlist) since that is where the
substance is. For high-engagement HN items with many comments, you may
also fetch the HN discussion page for expert context. Use judgment.

If a fetch fails or returns an error, judge based on the snippet alone
or skip the item.
"""


TOOL_NAME = "fetch_full_article"
TOOL_DESCRIPTION = (
    "Fetch the plain text of an article URL. Use when the snippet for "
    "an item is too thin to judge it or write a substantive summary. "
    "Returns the article text (HTML stripped, capped at "
    f"~{MAX_RETURN_CHARS} characters), or a string starting with "
    "'Error:' if the fetch fails or the URL is not allowed."
)
TOOL_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Absolute http(s) URL of the article to fetch.",
        }
    },
    "required": ["url"],
}


@dataclass
class CurateResult:
    digest_markdown: str
    tool_calls: int
    items_input: int
    items_output: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int


@dataclass
class ProviderResult:
    """Return shape of every llm_<provider>.py adapter.

    Adapters normalize provider-specific API responses into this common
    shape. Cache token fields default to 0 for providers that don't
    surface caching info (e.g. Gemini in the current SDK).
    """
    text: str
    tool_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_creation_tokens: int
    hit_max_output_tokens: bool


# The static allowlist is built from `homepage` only (not `url`). The `url`
# field is the feed/API endpoint; for HN it resolves to hn.algolia.com which
# the model has no reason to fetch as an article. Homepage hosts are what
# the model will actually see in Item.url values.
def build_static_allowlist(sources: Iterable[Source]) -> set[str]:
    hosts: set[str] = set()
    for s in sources:
        host = urlparse(s.homepage).hostname
        if host:
            hosts.add(host.lower())
    return hosts


def build_runtime_allowlist(items: Iterable[Item]) -> set[str]:
    hosts: set[str] = set()
    for it in items:
        if it.linked_url:
            host = urlparse(it.linked_url).hostname
            if host:
                hosts.add(host.lower())
    return hosts


def make_fetch_full_article(
    static_hosts: set[str],
    runtime_hosts: set[str],
    state: dict,
):
    """Return a tool function that closes over the allowlist and call counter."""
    allowlist = static_hosts | runtime_hosts

    def fetch_full_article(url: str) -> str:
        if state["calls"] >= TOOL_CALL_CAP:
            return f"Error: tool call cap reached ({TOOL_CALL_CAP} per run)"
        state["calls"] += 1

        if not isinstance(url, str) or not url.strip():
            return "Error: missing or invalid url argument"

        try:
            parsed = urlparse(url)
        except Exception as e:
            return f"Error: could not parse URL: {e}"

        if parsed.scheme not in ("http", "https"):
            return f"Error: scheme not allowed: {parsed.scheme!r}"

        host = (parsed.hostname or "").lower()
        if not host:
            return "Error: missing hostname in URL"
        if host not in allowlist:
            return f"Error: hostname not in allowlist: {host}"

        session = requests.Session()
        session.max_redirects = MAX_REDIRECTS

        try:
            with session.get(
                url,
                timeout=HTTP_TIMEOUT_SECONDS,
                stream=True,
                allow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            ) as resp:
                final_host = (urlparse(resp.url).hostname or "").lower()
                if final_host not in allowlist:
                    return (
                        f"Error: redirect ended at non-allowlisted host: "
                        f"{final_host}"
                    )

                if resp.status_code >= 400:
                    return f"Error: HTTP {resp.status_code}"

                content_type = (resp.headers.get("content-type") or "").lower()
                if not (
                    content_type.startswith("text/html")
                    or content_type.startswith("text/plain")
                ):
                    return f"Error: unsupported content type: {content_type or '(none)'}"

                buf = bytearray()
                for chunk in resp.iter_content(chunk_size=8192):
                    if not chunk:
                        continue
                    buf.extend(chunk)
                    if len(buf) > MAX_BYTES:
                        return f"Error: response exceeded {MAX_BYTES} bytes"

                encoding = resp.encoding or "utf-8"
                text = bytes(buf).decode(encoding, errors="replace")

                if content_type.startswith("text/html"):
                    soup = BeautifulSoup(text, "html.parser")
                    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                        tag.decompose()
                    text = soup.get_text(separator="\n")
                    text = re.sub(r"\n[ \t]*\n+", "\n\n", text).strip()

                if len(text) > MAX_RETURN_CHARS:
                    text = text[:MAX_RETURN_CHARS] + f"\n\n[truncated at {MAX_RETURN_CHARS} chars]"

                return text or "Error: empty response body"
        except requests.exceptions.TooManyRedirects:
            return f"Error: too many redirects (>{MAX_REDIRECTS})"
        except requests.exceptions.Timeout:
            return f"Error: request timed out after {HTTP_TIMEOUT_SECONDS}s"
        except requests.RequestException as e:
            return f"Error: fetch failed: {e}"

    return fetch_full_article


def build_user_message(items: list[Item], today: str) -> str:
    parts = [
        f"Today is {today}.",
        "",
        f"There are {len(items)} items below from the last 24 to 48 hours, "
        "after URL-level dedup against previously covered items. Each item "
        "has a source, title, URL, publish date, and a content snippet "
        "(which may be empty for some sources; use fetch_full_article if "
        "you need more).",
        "",
        "Items:",
        "",
    ]
    for i, it in enumerate(items, start=1):
        parts.append(f"[{i}] {it.source_name} | {it.published or 'no date'}")
        parts.append(f"Title: {it.title}")
        parts.append(f"URL: {it.url}")
        if it.linked_url:
            parts.append(f"Linked: {it.linked_url}")
        parts.append(f"Snippet: {it.content_snippet or '(empty)'}")
        parts.append("")
    return "\n".join(parts)


def _demote_stray_heading_bodies(markdown: str) -> str:
    """Strip `## ` / `# ` prefixes from lines that follow an item title.

    The model occasionally emits an item summary as `## body text…`
    instead of a plain paragraph, which markdown then renders as <h2>.
    This catches the deterministic pattern: a heading-prefixed line
    immediately after a `### [Title](url)` item title is body text.
    """
    lines = markdown.splitlines()
    item_title = re.compile(r"^###\s+\[.+\]\(.+\)\s*$")
    stray = re.compile(r"^#{1,2}\s+(?!\[)(.+)$")
    for i in range(1, len(lines)):
        if item_title.match(lines[i - 1]):
            m = stray.match(lines[i])
            if m:
                lines[i] = m.group(1)
    return "\n".join(lines)


def _count_items_in_digest(markdown: str) -> int:
    # Tolerate minor heading-level drift (##, ###, ####) so the count
    # stays meaningful if the model produces ## or #### instead of ###.
    return sum(1 for line in markdown.splitlines() if re.match(r"^#{2,4}\s", line))


def _empty_result() -> CurateResult:
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


def _select_provider_adapter():
    """Lazy import keeps each provider's SDK optional at install time."""
    if config.LLM_PROVIDER == "anthropic":
        from providers.anthropic import run_tool_use_loop
        return run_tool_use_loop
    if config.LLM_PROVIDER == "gemini":
        from providers.gemini import run_tool_use_loop
        return run_tool_use_loop
    raise RuntimeError(f"unknown LLM_PROVIDER: {config.LLM_PROVIDER!r}")


def curate(
    items: list[Item],
    sources: list[Source],
    *,
    today: str | None = None,
) -> CurateResult:
    if today is None:
        today = datetime.now(timezone.utc).date().isoformat()

    if not items:
        return _empty_result()

    static_hosts = build_static_allowlist(sources)
    runtime_hosts = build_runtime_allowlist(items)
    state = {"calls": 0}
    fetch_tool = make_fetch_full_article(static_hosts, runtime_hosts, state)

    run_tool_use_loop = _select_provider_adapter()

    result = run_tool_use_loop(
        model=config.MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_message=build_user_message(items, today),
        tool_name=TOOL_NAME,
        tool_description=TOOL_DESCRIPTION,
        tool_input_schema=TOOL_INPUT_SCHEMA,
        tool_handler=fetch_tool,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    digest_md = _demote_stray_heading_bodies(result.text)
    if result.hit_max_output_tokens:
        digest_md += (
            "\n\n[Digest truncated — output exceeded max_tokens limit. "
            "Some items may be missing.]"
        )

    return CurateResult(
        digest_markdown=digest_md,
        tool_calls=result.tool_calls,
        items_input=len(items),
        items_output=_count_items_in_digest(digest_md),
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_creation_tokens=result.cache_creation_tokens,
    )