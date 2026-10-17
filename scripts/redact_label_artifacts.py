"""Pseudonymise individual accounts in the published label artifacts.

The brief requires that participant privacy is protected, and the ethics
document claims that no output names an individual. The stratified sample drew
14 tweets, and its key file carried the account handle of each one beside a
political statement. Several of those accounts are private individuals.

Nothing in the analysis needs the handle: concordance, agreement and the
benchmark are all computed by SOURCE TYPE. So each tweet account becomes a
stable pseudonym, and the mapping stays out of the repository.

Handles inside an item's text are redacted the same way, because a mention
names a person as surely as a byline does.
"""

from __future__ import annotations

import hashlib
import pathlib
import re

import pandas as pd

LABELS = pathlib.Path("outputs/labels")
MAPPING = pathlib.Path("outputs/labels_private/account_pseudonyms.csv")
MENTION = re.compile(r"@([A-Za-z0-9_]{2,})")


def pseudonym(handle: str) -> str:
    """A stable, non-reversible pseudonym for one account."""
    digest = hashlib.sha256(f"perceptual-bias:{handle}".encode()).hexdigest()[:8]
    return f"account_{digest}"


def main() -> int:
    key_path = LABELS / "coding_sample_key.csv"
    key = pd.read_csv(key_path)
    tweets = key["corpus"] == "tweets"
    handles = sorted(key.loc[tweets, "source"].dropna().unique())

    mapping = {handle: pseudonym(handle) for handle in handles}
    key.loc[tweets, "source"] = key.loc[tweets, "source"].map(
        lambda value: mapping.get(value, value)
    )
    key.to_csv(key_path, index=False)

    MAPPING.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        sorted(mapping.items()), columns=["handle", "pseudonym"]
    ).to_csv(MAPPING, index=False)

    # Mentions inside any published text column.
    redacted_cells = 0
    for path in sorted(LABELS.glob("*.csv")):
        frame = pd.read_csv(path)
        changed = False
        for column in frame.columns:
            # pandas 3 reads text as the 'str' dtype, not 'object'. A check
            # against `object` alone silently skipped every text column, which
            # is how the first run reported zero redactions on a file that
            # contained a handle.
            if not pd.api.types.is_string_dtype(frame[column]):
                continue
            original = frame[column]
            filled = original.fillna("")
            after = filled.str.replace(MENTION, "@[handle]", regex=True)
            hits = int((filled != after).sum())
            if hits:
                # Put the redacted text back only where there was text. A cell
                # that was empty stays empty: this file is the unlabelled
                # instrument, and writing "" into an annotator column would
                # make an empty label look like a given one.
                frame[column] = after.where(original.notna(), original)
                redacted_cells += hits
                changed = True
        if changed:
            frame.to_csv(path, index=False)

    print(f"pseudonymised {len(mapping)} accounts in {key_path.name}")
    print(f"redacted {redacted_cells} mention(s) inside published text")
    print(f"mapping written to {MAPPING} (kept out of git)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
