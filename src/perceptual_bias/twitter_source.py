"""Budget-capped twitterapi.io adapter.

The account has a small, finite credit balance and **one request costs 300
credits regardless of how many tweets come back** (measured, not assumed: a
single probe moved the balance by 0.3k for 20 tweets). Everything here is built
around that fact:

* a **persisted ledger** counts credits across runs, so a crash or a re-run
  cannot silently reset the budget and overspend;
* every page is **cached to disk by (month, page)**, so re-running the pipeline,
  rebuilding the report or reloading the dashboard costs nothing;
* the cap is checked *before* each request, never after;
* failures abort immediately rather than retrying, because a retry loop against
  a paid endpoint is how a balance disappears.

The econometrics never depend on this source. GDELT carries the monthly series;
tweets provide term loadings and a social-vs-news tone contrast.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.twitterapi.io/twitter/tweet/advanced_search"

# Measured from the probe: one advanced_search call = 300 credits, 20 tweets.
CREDITS_PER_REQUEST = 300
TWEETS_PER_PAGE = 20

# Quality filters from the project spec.
MIN_FOLLOWERS = 10
MIN_VIEWS = 100

DEFAULT_TERMS = (
    '(inflation OR "cost of living" OR economy OR groceries OR "gas prices" '
    'OR recession OR layoffs OR "grocery prices" OR rent)'
)


class BudgetExceeded(RuntimeError):
    """Raised instead of making a request that would breach the cap."""


class TwitterError(RuntimeError):
    """Raised when the API responds with anything other than a usable payload."""


@dataclass
class CreditLedger:
    """Credits spent, persisted so the cap survives restarts."""

    path: pathlib.Path
    cap: int
    spent: int = 0
    requests_made: int = 0

    @classmethod
    def load(cls, path: pathlib.Path, cap: int) -> "CreditLedger":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                path=path,
                cap=cap,
                spent=int(data.get("spent", 0)),
                requests_made=int(data.get("requests_made", 0)),
            )
        return cls(path=path, cap=cap)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {
                    "spent": self.spent,
                    "requests_made": self.requests_made,
                    "cap": self.cap,
                    "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @property
    def remaining(self) -> int:
        return max(self.cap - self.spent, 0)

    def check(self, cost: int = CREDITS_PER_REQUEST) -> None:
        if self.spent + cost > self.cap:
            raise BudgetExceeded(
                f"request would cost {cost}, spent {self.spent} of cap {self.cap}"
            )

    def charge(self, cost: int = CREDITS_PER_REQUEST) -> None:
        self.spent += cost
        self.requests_made += 1
        self.save()


@dataclass
class PullResult:
    """What a pull produced, and what it cost."""

    tweets: pd.DataFrame
    requests_made: int
    credits_spent: int
    cached_pages: int
    months: tuple[str, ...] = field(default_factory=tuple)

    def summary(self) -> str:
        return (
            f"{len(self.tweets)} tweets over {len(self.months)} months | "
            f"{self.requests_made} paid requests ({self.credits_spent} credits), "
            f"{self.cached_pages} pages served from cache"
        )


def day_query(day: str, terms: str = DEFAULT_TERMS) -> str:
    """Build a single-day advanced-search query for ``YYYY-MM-DD``.

    **Use this, not `month_query`.** A month-wide window with
    ``queryType: Latest`` returns only the newest matching tweets, which for a
    high-volume economic query is roughly fourteen minutes of traffic at the
    end of the window — measured, after spending 4 200 credits discovering it.
    A one-day window with ``queryType: Top`` returns tweets inside the
    requested day that pass the quality filters.
    """
    start = pd.Timestamp(day)
    end = start + pd.Timedelta(days=1)
    return f"{terms} lang:en -is:retweet since:{start:%Y-%m-%d} until:{end:%Y-%m-%d}"


def month_query(month: str, terms: str = DEFAULT_TERMS) -> str:
    """Build a date-bounded advanced-search query for ``YYYY-MM``.

    The endpoint treats ``until:`` loosely, so results are clamped again on the
    client side after parsing.
    """
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthEnd(0)
    return (
        f"{terms} lang:en -is:retweet "
        f"since:{start:%Y-%m-%d} until:{(end + pd.Timedelta(days=1)):%Y-%m-%d}"
    )


def _parse_tweet(raw: dict) -> dict | None:
    """Flatten one API tweet, returning None if it fails the quality filters."""
    author = raw.get("author") or {}
    followers = author.get("followers")
    if followers is None:
        followers = author.get("followersCount", 0)
    views = raw.get("viewCount", 0) or 0
    try:
        followers = int(followers or 0)
        views = int(views)
    except (TypeError, ValueError):
        return None
    if followers < MIN_FOLLOWERS or views < MIN_VIEWS:
        return None
    text = (raw.get("text") or "").strip()
    if not text:
        return None
    created_raw = raw.get("createdAt") or raw.get("created_at")
    try:
        created = pd.to_datetime(created_raw, format="%a %b %d %H:%M:%S %z %Y", utc=True)
    except (ValueError, TypeError):
        created = pd.to_datetime(created_raw, utc=True, errors="coerce")
    if pd.isna(created):
        return None
    return {
        "id": raw.get("id"),
        "created_at": created,
        "text": text,
        "lang": raw.get("lang"),
        "views": views,
        "likes": int(raw.get("likeCount") or 0),
        "retweets": int(raw.get("retweetCount") or 0),
        "followers": followers,
        "username": author.get("userName"),
    }


class TwitterSource:
    """Cached, budget-capped client for one endpoint and one purpose."""

    def __init__(
        self,
        cache_dir: pathlib.Path,
        api_key: str | None = None,
        cap: int = 5000,
    ) -> None:
        self.cache_dir = pathlib.Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("TWITTERAPI_IO_KEY", "").strip()
        self.ledger = CreditLedger.load(self.cache_dir / "_ledger.json", cap=cap)

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _cache_path(self, month: str, page: int) -> pathlib.Path:
        return self.cache_dir / f"{month}_p{page:02d}.json"

    def _fetch_page(self, query: str, cursor: str | None) -> dict:
        self.ledger.check()
        params = {"query": query, "queryType": "Latest"}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(
            ENDPOINT, headers={"X-API-Key": self.api_key}, params=params, timeout=45
        )
        # Charge as soon as the call is made: a non-200 may still have cost.
        self.ledger.charge()
        if response.status_code != 200:
            raise TwitterError(f"HTTP {response.status_code}: {response.text[:200]}")
        try:
            return response.json()
        except ValueError as exc:
            raise TwitterError(f"non-JSON response: {exc}") from exc

    def pull_month(self, month: str, max_pages: int, terms: str = DEFAULT_TERMS) -> tuple[list[dict], int, int]:
        """Fetch up to `max_pages` pages for one month, using cache where possible.

        Returns (tweets, paid_requests, cached_pages).
        """
        query = month_query(month, terms=terms)
        collected: list[dict] = []
        cursor: str | None = None
        paid = 0
        cached = 0

        for page in range(1, max_pages + 1):
            cache_path = self._cache_path(month, page)
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                cached += 1
            else:
                if not self.configured:
                    logger.warning("no twitterapi.io key; stopping %s at page %d", month, page)
                    break
                try:
                    payload = self._fetch_page(query, cursor)
                except BudgetExceeded as exc:
                    logger.warning("budget stop on %s page %d: %s", month, page, exc)
                    break
                cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                paid += 1

            raw_tweets = payload.get("tweets") or []
            collected.extend(raw_tweets)
            cursor = payload.get("next_cursor")
            if not payload.get("has_next_page") or not cursor:
                break

        return collected, paid, cached

    def pull_days(
        self,
        days: tuple[str, ...],
        pages_per_day: int = 1,
        terms: str = DEFAULT_TERMS,
    ) -> PullResult:
        """Sample specific days with `queryType: Top`.

        This is the method that works. One request per day returns tweets that
        are inside the requested day and about the economy. Sampling days
        across many months builds a real time series, which a month-wide
        request cannot do.
        """
        rows: list[dict] = []
        paid_total = 0
        cached_total = 0
        before = self.ledger.spent

        for day in days:
            query = day_query(day, terms=terms)
            cursor: str | None = None
            for page in range(1, pages_per_day + 1):
                cache_path = self.cache_dir / f"day_{day}_p{page:02d}.json"
                if cache_path.exists():
                    payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    cached_total += 1
                else:
                    if not self.configured:
                        logger.warning("no twitterapi.io key; stopping at %s", day)
                        break
                    try:
                        payload = self._fetch_page_top(query, cursor)
                    except BudgetExceeded as exc:
                        logger.warning("budget stop at %s: %s", day, exc)
                        return self._finalise(rows, paid_total, cached_total, days, before)
                    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    paid_total += 1

                window_start = pd.Timestamp(day, tz="UTC")
                window_end = window_start + pd.Timedelta(days=1)
                kept = 0
                for item in payload.get("tweets") or []:
                    parsed = _parse_tweet(item)
                    if parsed is None:
                        continue
                    if not (window_start <= parsed["created_at"] < window_end):
                        continue
                    parsed["day"] = day
                    parsed["month"] = day[:7]
                    rows.append(parsed)
                    kept += 1
                logger.info("%s page %d: %d kept", day, page, kept)
                cursor = payload.get("next_cursor")
                if not payload.get("has_next_page") or not cursor:
                    break

        return self._finalise(rows, paid_total, cached_total, days, before)

    def _finalise(
        self,
        rows: list[dict],
        paid: int,
        cached: int,
        labels: tuple[str, ...],
        before: int,
    ) -> PullResult:
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.drop_duplicates(subset="id").sort_values("created_at").reset_index(drop=True)
        return PullResult(
            tweets=frame,
            requests_made=paid,
            credits_spent=self.ledger.spent - before,
            cached_pages=cached,
            months=tuple(labels),
        )

    def _fetch_page_top(self, query: str, cursor: str | None) -> dict:
        """Fetch one page with `queryType: Top` (the mode that respects the day)."""
        self.ledger.check()
        params = {"query": query, "queryType": "Top"}
        if cursor:
            params["cursor"] = cursor
        response = requests.get(
            ENDPOINT, headers={"X-API-Key": self.api_key}, params=params, timeout=45
        )
        self.ledger.charge()
        if response.status_code != 200:
            raise TwitterError(f"HTTP {response.status_code}: {response.text[:200]}")
        try:
            return response.json()
        except ValueError as exc:
            raise TwitterError(f"non-JSON response: {exc}") from exc

    def pull(self, months: tuple[str, ...], pages_per_month: int, terms: str = DEFAULT_TERMS) -> PullResult:
        """Pull several months and return a filtered, deduplicated frame."""
        rows: list[dict] = []
        paid_total = 0
        cached_total = 0
        before = self.ledger.spent

        for month in months:
            raw, paid, cached = self.pull_month(month, pages_per_month, terms=terms)
            paid_total += paid
            cached_total += cached
            window_start = pd.Timestamp(f"{month}-01", tz="UTC")
            window_end = window_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
            kept = 0
            for item in raw:
                parsed = _parse_tweet(item)
                if parsed is None:
                    continue
                # Clamp: the endpoint honours `until:` loosely.
                if not (window_start <= parsed["created_at"] < window_end):
                    continue
                parsed["month"] = month
                rows.append(parsed)
                kept += 1
            logger.info("%s: %d raw -> %d kept after filters", month, len(raw), kept)

        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.drop_duplicates(subset="id").sort_values("created_at").reset_index(drop=True)

        return PullResult(
            tweets=frame,
            requests_made=paid_total,
            credits_spent=self.ledger.spent - before,
            cached_pages=cached_total,
            months=tuple(months),
        )
