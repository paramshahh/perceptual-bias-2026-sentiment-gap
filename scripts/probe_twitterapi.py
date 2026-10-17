"""One-request probe of the twitterapi.io advanced_search endpoint.

Deliberately minimal and deliberately single-shot: it makes exactly ONE HTTP
request, never retries, and writes the raw response to the cache so the probe
itself never has to be paid for twice.

Purpose is to learn three things before committing any real budget:
  1. does the key authenticate,
  2. what does the response actually look like (field names, nesting),
  3. what does one call cost, and what balance remains.

Run:  .venv/bin/python scripts/probe_twitterapi.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import requests
from dotenv import load_dotenv

ENDPOINT = "https://api.twitterapi.io/twitter/tweet/advanced_search"
CACHE_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "tweets"

# Narrow two-day window inside the worst-gap month (2026-05, gap -33.13).
# Quality filters per the project spec: no retweets, English only.
PROBE_QUERY = "(inflation OR economy OR prices) lang:en -is:retweet since:2026-05-10 until:2026-05-12"


def main() -> int:
    load_dotenv()
    key = os.environ.get("TWITTERAPI_IO_KEY", "").strip()
    if not key:
        print("TWITTERAPI_IO_KEY not set; nothing spent.")
        return 1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    params = {"query": PROBE_QUERY, "queryType": "Latest"}

    print(f"ONE request to {ENDPOINT}")
    print(f"query: {PROBE_QUERY}")
    try:
        response = requests.get(
            ENDPOINT, headers={"X-API-Key": key}, params=params, timeout=45
        )
    except requests.RequestException as exc:
        print(f"request failed, nothing charged beyond the attempt: {exc}")
        return 1

    print(f"HTTP {response.status_code}")

    # Credit/quota information is conventionally returned in headers; print any
    # header that looks relevant so the real per-call cost is known.
    interesting = {
        name: value
        for name, value in response.headers.items()
        if any(token in name.lower() for token in ("credit", "quota", "limit", "balance", "usage", "cost"))
    }
    print("quota-ish headers:", interesting or "(none)")

    if response.status_code != 200:
        print("body (first 400 chars):", response.text[:400])
        return 1

    try:
        payload = response.json()
    except ValueError:
        print("non-JSON body (first 400 chars):", response.text[:400])
        return 1

    cache_path = CACHE_DIR / "probe_2026-05-10.json"
    cache_path.write_text(json.dumps(payload, indent=2)[:2_000_000], encoding="utf-8")

    print("top-level keys:", list(payload)[:12])
    tweets = payload.get("tweets") or payload.get("data") or []
    print(f"tweets returned: {len(tweets)}")
    print("has_next_page:", payload.get("has_next_page"))

    if tweets:
        sample = tweets[0]
        print("tweet fields:", sorted(sample)[:20])
        for field in ("createdAt", "created_at", "viewCount", "likeCount", "retweetCount"):
            if field in sample:
                print(f"  {field}: {sample[field]}")
        author = sample.get("author") or {}
        if isinstance(author, dict):
            print("  author fields:", sorted(author)[:12])
            for field in ("followers", "followersCount", "userName"):
                if field in author:
                    print(f"    {field}: {author[field]}")
        text = (sample.get("text") or "")[:160].replace("\n", " ")
        print(f"  sample text: {text}")

    print(f"cached -> {cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
