"""Pull a sampled GDELT headline corpus for the media hypothesis.

The request pattern follows the practice in Ken Blake's "GDELT Headline Fetch"
(rpubs.com/drkblake/1007551, 2023-09-07):

* `mode=artlist`, `maxrecords=250`, `sort=datedesc`;
* `format=CSV`, which gives a smaller answer than JSON and parses directly
  into a table with `URL, MobileURL, Date, Title, SourceCountry, Language`;
* `startdatetime=YYYYMMDD000000` and `enddatetime=YYYYMMDD235959`, which is the
  same calendar day. Do not use the next day at 000000: that includes one
  second of the following day and changes which articles the sort returns;
* a pause between requests.

That script uses a two-second pause. We use eight seconds, because GDELT asks
for one request each five seconds and because this project already caused a
sustained block with fast probe requests. The block is caused by the quantity
of fast requests, not by one request.

The GDELT API holds data from January 2017.

We do not restrict the outlets with `domainis:`. The hypothesis is about the
information environment of a household, not about a list of publications, so
the corpus covers all United States English media.

Each day is cached. A second run costs no network.

Run:  .venv/bin/python scripts/pull_gdelt.py [start_month] [end_month]
"""

from __future__ import annotations

import io
import pathlib
import sys
import time

import pandas as pd
import requests

API = "https://api.gdeltproject.org/api/v2/doc/doc"
CACHE = pathlib.Path("data/raw/gdelt")
HEADERS = {"User-Agent": "perceptual-bias-research/1.0 (academic project)"}

TERMS = (
    "inflation",
    '"cost of living"',
    '"grocery prices"',
    '"gas prices"',
    "recession",
    "layoffs",
    '"consumer sentiment"',
    "economy",
)
QUERY = "(" + " OR ".join(TERMS) + ") sourcelang:english sourcecountry:US"

MIN_INTERVAL = 8.0
MAX_ATTEMPTS = 5
SAMPLE_DAYS = (12, 24)
ARCHIVE_START = pd.Timestamp("2017-01-01")


def fetch_day(day: pd.Timestamp, state: dict) -> pd.DataFrame | None:
    """Fetch the headlines of one day. Returns a table, or None on failure."""
    cache_path = CACHE / f"{day:%Y%m%d}.csv"
    if cache_path.exists():
        state["cached"] += 1
        try:
            return pd.read_csv(cache_path)
        except Exception:
            cache_path.unlink()  # a damaged cache file must not stop the run

    params = {
        "query": QUERY,
        "mode": "artlist",
        "maxrecords": 250,
        "sort": "datedesc",
        "startdatetime": f"{day:%Y%m%d}000000",
        "enddatetime": f"{day:%Y%m%d}235959",
        "format": "CSV",
    }

    wait = MIN_INTERVAL
    for attempt in range(1, MAX_ATTEMPTS + 1):
        elapsed = time.monotonic() - state["last_request"]
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)
        try:
            response = requests.get(API, params=params, headers=HEADERS, timeout=60)
        except requests.RequestException as exc:
            print(f"  {day:%Y-%m-%d} attempt {attempt}: network error {exc}", flush=True)
            time.sleep(wait)
            wait *= 2
            continue
        finally:
            state["last_request"] = time.monotonic()

        if response.status_code == 429:
            print(f"  {day:%Y-%m-%d} attempt {attempt}: 429, wait {wait:.0f}s", flush=True)
            time.sleep(wait)
            wait *= 2
            continue
        if response.status_code != 200:
            print(f"  {day:%Y-%m-%d}: HTTP {response.status_code}", flush=True)
            return None

        text = response.text
        if not text.strip() or text.lstrip().startswith("<"):
            print(f"  {day:%Y-%m-%d}: not CSV: {text[:90]}", flush=True)
            return None
        try:
            frame = pd.read_csv(io.StringIO(text))
        except Exception as exc:
            print(f"  {day:%Y-%m-%d}: CSV parse failed ({exc})", flush=True)
            return None

        cache_path.write_text(text, encoding="utf-8")
        state["fetched"] += 1
        return frame

    print(f"  {day:%Y-%m-%d}: gave up after {MAX_ATTEMPTS} attempts", flush=True)
    return None


def main() -> int:
    start = sys.argv[1] if len(sys.argv) > 1 else "2017-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08"
    CACHE.mkdir(parents=True, exist_ok=True)

    months = pd.date_range(f"{start}-01", f"{end}-01", freq="MS")
    days = [
        month + pd.Timedelta(days=offset - 1)
        for month in months
        for offset in SAMPLE_DAYS
    ]
    days = [day for day in days if day >= ARCHIVE_START]
    print(f"days: {len(days)}, {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d}")
    print(f"time if nothing is cached: {len(days) * MIN_INTERVAL / 60:.0f} min")

    state = {"last_request": 0.0, "cached": 0, "fetched": 0}
    frames: list[pd.DataFrame] = []
    empty: list[str] = []

    for index, day in enumerate(days, start=1):
        frame = fetch_day(day, state)
        if frame is None or frame.empty:
            empty.append(f"{day:%Y-%m-%d}")
            continue
        frame = frame.copy()
        frame["sample_day"] = f"{day:%Y-%m-%d}"
        frames.append(frame)
        if index % 20 == 0:
            rows = sum(len(f) for f in frames)
            print(f"  {index}/{len(days)} days, {rows} headlines", flush=True)

    if not frames:
        print("no data retrieved")
        return 1

    corpus = pd.concat(frames, ignore_index=True)
    corpus.columns = [str(c).strip().lower() for c in corpus.columns]
    if "url" in corpus:
        corpus = corpus.drop_duplicates(subset="url")
    date_column = "date" if "date" in corpus else corpus.columns[0]
    corpus["date"] = pd.to_datetime(corpus[date_column], errors="coerce", format="mixed")
    corpus = corpus.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    corpus["month"] = corpus["date"].dt.to_period("M").astype(str)

    out = CACHE / "headlines.parquet"
    corpus.to_parquet(out)

    print()
    print(f"headlines: {len(corpus)}")
    print(f"days fetched: {state['fetched']}, cached: {state['cached']}, empty: {len(empty)}")
    print(f"months covered: {corpus['month'].nunique()}")
    print(f"columns: {list(corpus.columns)}")
    print(f"saved -> {out}")
    if empty:
        print(f"empty days: {', '.join(empty[:12])}{'...' if len(empty) > 12 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
