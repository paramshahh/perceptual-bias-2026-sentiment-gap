"""Pull an economic news corpus from the GDELT GKG bulk files.

**Why this exists.** The GDELT DOC API gives headlines, which is the better
input for the TF-IDF matrix. But the API permits one request each five seconds
and it put this project in a sustained block. The GKG bulk files are on a plain
file server with no rate limit. GDELT's own 429 message directs heavy users
there.

**What this gives, and what it costs.** Each GKG record carries the themes that
GDELT assigns to an article and a tone vector that GDELT computes. It does not
carry the headline. So:

* `media_tone` comes from GDELT's own measurement, not from our lexicon. This
  is arguably the better measurement, because it is independent of us.
* The TF-IDF matrix and the SVD use **theme tags**, not English words. Table 2
  then shows topic tags such as `ECON_INFLATION`, not words.

GKG publishes one file each 15 minutes. We take two files for each sampled day.
A file holds articles from the whole world, so we keep only United States
sources with an economic theme.

Column positions in GKG 2.1 (tab separated, 27 columns):

    1  date (YYYYMMDDHHMMSS)      15  V2.1 tone vector
    3  source common name         23  all names
    4  document URL                7  V1 themes

The tone vector is
`tone,positive,negative,polarity,activity_density,self_density,word_count`.
We keep the first three fields.

Run:  .venv/bin/python scripts/pull_gkg.py [start_month] [end_month]
"""

from __future__ import annotations

import csv
import io
import pathlib
import sys
import time
import zipfile

import pandas as pd
import requests

# GKG records carry very large fields (the GCAM vector, the enhanced theme
# list). Python's default CSV field limit of 131072 characters stops the read
# with "field larger than field limit". Raise it before any file is parsed.
csv.field_size_limit(10_000_000)

BASE = "http://data.gdeltproject.org/gdeltv2"
CACHE = pathlib.Path("data/raw/gkg")
HEADERS = {"User-Agent": "perceptual-bias-research/1.0 (academic project)"}

# Two files for each day, spread over the news cycle.
FILE_TIMES = ("120000", "200000")
SAMPLE_DAYS = (12, 24)
ARCHIVE_START = pd.Timestamp("2017-01-01")
PAUSE = 1.0  # a courtesy pause; the file server sets no limit

ECONOMIC_THEMES = (
    "ECON_INFLATION",
    "ECON_STOCKMARKET",
    "ECON_INTEREST_RATE",
    "ECON_UNEMPLOYMENT",
    "ECON_COST_OF_LIVING",
    "ECON_DEBT",
    "ECON_PRICECONTROL",
    "ECON_RECESSION",
    "ECON_BANKRUPTCY",
    "ECON_HOUSING",
    "UNEMPLOYMENT",
    "LAYOFF",
    "TAX_",
    "WB_1216_ECONOMIC",
    "WB_450_ECONOMIC",
)

DATE_COL, SOURCE_COL, URL_COL, THEMES_COL, TONE_COL = 1, 3, 4, 7, 15
MAX_COL = 23


def is_economic(themes: str) -> bool:
    return any(theme in themes for theme in ECONOMIC_THEMES)


def fetch_file(stamp: str, state: dict) -> list[dict]:
    """Download and filter one 15-minute GKG file."""
    cache_path = CACHE / f"{stamp}.parquet"
    if cache_path.exists():
        state["cached"] += 1
        try:
            return pd.read_parquet(cache_path).to_dict("records")
        except Exception:
            cache_path.unlink()

    url = f"{BASE}/{stamp}.gkg.csv.zip"
    try:
        response = requests.get(url, headers=HEADERS, timeout=120)
    except requests.RequestException as exc:
        print(f"  {stamp}: network error {exc}", flush=True)
        return []
    if response.status_code != 200:
        print(f"  {stamp}: HTTP {response.status_code}", flush=True)
        return []

    try:
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        name = archive.namelist()[0]
        stream = io.TextIOWrapper(archive.open(name), encoding="utf-8", errors="replace")
        reader = csv.reader(stream, delimiter="\t")
    except Exception as exc:
        print(f"  {stamp}: unreadable archive ({exc})", flush=True)
        return []

    rows: list[dict] = []
    for record in reader:
        if len(record) <= MAX_COL:
            continue
        themes = record[THEMES_COL]
        if not themes or not is_economic(themes):
            continue
        url_value = record[URL_COL]
        if not url_value:
            continue
        tone_parts = (record[TONE_COL] or "").split(",")
        try:
            tone = float(tone_parts[0])
            positive = float(tone_parts[1])
            negative = float(tone_parts[2])
        except (ValueError, IndexError):
            continue
        rows.append(
            {
                "stamp": record[DATE_COL],
                "source": record[SOURCE_COL],
                "url": url_value,
                "themes": themes,
                "tone": tone,
                "positive": positive,
                "negative": negative,
            }
        )

    if rows:
        pd.DataFrame(rows).to_parquet(cache_path)
    state["fetched"] += 1
    state["bytes"] += len(response.content)
    return rows


def main() -> int:
    start = sys.argv[1] if len(sys.argv) > 1 else "2017-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2026-08"
    CACHE.mkdir(parents=True, exist_ok=True)

    months = pd.date_range(f"{start}-01", f"{end}-01", freq="MS")
    days = [m + pd.Timedelta(days=d - 1) for m in months for d in SAMPLE_DAYS]
    days = [d for d in days if d >= ARCHIVE_START]
    stamps = [f"{day:%Y%m%d}{t}" for day in days for t in FILE_TIMES]
    print(f"files: {len(stamps)} ({len(days)} days x {len(FILE_TIMES)})")
    print(f"window: {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}")

    state = {"cached": 0, "fetched": 0, "bytes": 0}
    rows: list[dict] = []
    for index, stamp in enumerate(stamps, start=1):
        rows.extend(fetch_file(stamp, state))
        if PAUSE:
            time.sleep(PAUSE)
        if index % 25 == 0:
            print(
                f"  {index}/{len(stamps)} files, {len(rows)} economic articles, "
                f"{state['bytes']/1e9:.2f} GB downloaded",
                flush=True,
            )

    if not rows:
        print("no rows retrieved")
        return 1

    frame = pd.DataFrame(rows).drop_duplicates(subset="url")
    frame["date"] = pd.to_datetime(frame["stamp"], format="%Y%m%d%H%M%S", errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame["month"] = frame["date"].dt.to_period("M").astype(str)

    out = CACHE / "gkg_economic.parquet"
    frame.to_parquet(out)
    print()
    print(f"articles: {len(frame)}")
    print(f"months covered: {frame['month'].nunique()}")
    print(f"files fetched: {state['fetched']}, cached: {state['cached']}")
    print(f"downloaded: {state['bytes']/1e9:.2f} GB")
    print(f"mean tone: {frame['tone'].mean():.3f}")
    print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
