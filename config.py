"""Tuning constants for the news digest agent.

Centralized so things like the User-Agent string or model ID get
updated in one place. No environment-variable overrides — these are
tuning constants, not deployment config.

What lives elsewhere on purpose:
  - Source URLs / dispatch kinds    sources.py
  - HN keyword filter list          fetchers/hn.py (long, structural,
                                    belongs near the consumer)
  - System prompt + tool definition curator.py (code-adjacent)
  - Cron schedule                   .github/workflows/daily-digest.yml
"""

# Shared across all outbound HTTP fetchers
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
HTTP_TIMEOUT_SECONDS = 30
SNIPPET_CHARS = 1500

# Anthropic blog scraper
ANTHROPIC_BASE_URL = "https://www.anthropic.com"

# Hacker News fetcher
HN_LOOKBACK_HOURS = 48
HN_POINTS_FLOOR = 50
HN_HITS_PER_PAGE = 100

# Curator (LLM tool-use loop)
LLM_PROVIDER = "gemini"   # "anthropic" | "gemini"

PROVIDER_MODEL = {
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-3.1-flash-lite",
}
MODEL = PROVIDER_MODEL[LLM_PROVIDER]

MAX_OUTPUT_TOKENS = 8192
TOOL_CALL_CAP = 20

# fetch_full_article tool limits
MAX_BYTES = 1_000_000
MAX_RETURN_CHARS = 50_000
MAX_REDIRECTS = 5

# State management
SEEN_TTL_DAYS = 14
HEALTH_RUNS_KEPT = 14

# main.py
ITEM_AGE_LIMIT_DAYS = 7
