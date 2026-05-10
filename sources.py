"""Source registry.

Each entry's `kind` field decides which fetcher handles it:
    "rss"            -> fetchers.rss.fetch(source)
    "anthropic_blog" -> fetchers.anthropic_blog.fetch(source)
    "hn"             -> fetchers.hn.fetch(source)

Adding a new fetcher is mechanical: pick a new kind string, add a dispatch
case in main.py, write the fetcher.

Sources considered but not active:

    Import AI (dropped 2026-05-10)
        Substack sits behind Cloudflare and 403s the feed from GitHub
        Actions IP ranges regardless of User-Agent or other request
        headers. Coverage overlaps Interconnects, Latent Space, and
        Ahead of AI heavily, so the signal loss is small.

    Meta AI blog (dropped permanently)
        No RSS feed exists. Robots.txt explicitly blocks scrapers
        (Scrapy, PetalBot, etc.) and the site is JS-rendered behind
        a Facebook stack. Their major announcements echo through
        Tier 1 newsletters and Hacker News, so the signal loss is small.

    AlphaSignal (deferred to a future iteration)
        Email-newsletter only. Public site is a Next.js subscription
        landing page with no archive. Has unique value (research paper
        coverage thinner in our other sources), so a future iteration
        will add an "email" source kind to ingest it.

    The Batch by DeepLearning.AI (deferred to a future iteration)
        Page is React-rendered with no feed link tag. /feed/, /feed.xml,
        /rss/ all 404 or 500. Primarily a weekly email newsletter.
        May be added via the email path alongside AlphaSignal.

RSS URLs below were verified 2026-05-02.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Source:
    name: str
    tier: int
    kind: str   # "rss" | "anthropic_blog" | "hn"
    url: str    # feed URL for rss; index/API endpoint for custom kinds
    homepage: str


SOURCES: list[Source] = [
    # Tier 1: curated weekly digests
    Source(
        name="Latent Space",
        tier=1,
        kind="rss",
        url="https://www.latent.space/feed",
        homepage="https://www.latent.space",
    ),
    Source(
        name="Interconnects",
        tier=1,
        kind="rss",
        url="https://www.interconnects.ai/feed",
        homepage="https://www.interconnects.ai",
    ),
    Source(
        name="Ahead of AI",
        tier=1,
        kind="rss",
        url="https://magazine.sebastianraschka.com/feed",
        homepage="https://magazine.sebastianraschka.com",
    ),

    # Tier 2: lab primary sources
    Source(
        name="Anthropic",
        tier=2,
        kind="anthropic_blog",
        url="https://www.anthropic.com/news",
        homepage="https://www.anthropic.com/news",
    ),
    Source(
        name="OpenAI",
        tier=2,
        kind="rss",
        url="https://openai.com/news/rss.xml",
        homepage="https://openai.com/news",
    ),
    Source(
        name="Google DeepMind",
        tier=2,
        kind="rss",
        url="https://deepmind.google/blog/rss.xml",
        homepage="https://deepmind.google/blog",
    ),
    Source(
        name="Hugging Face",
        tier=2,
        kind="rss",
        url="https://huggingface.co/blog/feed.xml",
        homepage="https://huggingface.co/blog",
    ),

    # Tier 3: practitioner filters
    Source(
        name="TLDR AI",
        tier=3,
        kind="rss",
        url="https://tldr.tech/api/rss/ai",
        homepage="https://tldr.tech/ai",
    ),

    # Tier 4: community signal
    Source(
        name="Hacker News",
        tier=4,
        kind="hn",
        url="https://hn.algolia.com/api/v1/search_by_date",
        homepage="https://news.ycombinator.com",
    ),
]