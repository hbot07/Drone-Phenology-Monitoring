#!/usr/bin/env python3
"""Rename weekly-meeting decks and extracted Markdown in chronological order."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path


MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}


@dataclass(frozen=True)
class Entry:
    old_path: Path
    old_name: str
    parsed_date: date
    per_day_index: int
    new_name: str


def parse_deck_date(stem: str) -> date:
    iso_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})_", stem)
    if iso_match:
        year, month, day = (int(iso_match.group(i)) for i in range(1, 4))
        return date(year, month, day)

    clean = stem.replace("_", " ").replace("-", " ")
    clean = re.sub(r"\(\d+\)", " ", clean)
    clean = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", clean)
    clean = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", clean)
    clean = re.sub(r"\s+", " ", clean).strip().lower()
    parts = clean.split()

    day = None
    month = None
    year = None

    for part in parts:
        if part in MONTHS:
            month = MONTHS[part]
        elif part.isdigit():
            n = int(part)
            if n > 31:
                year = n
            elif day is None:
                day = n
            elif year is None:
                year = n

    if year is not None and year < 100:
        year += 2000

    if not (day and month and year):
        raise ValueError(f"Could not parse date from {stem!r} -> {parts!r}")

    return date(year, month, day)


def readable_slug(stem: str) -> str:
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}_", "", stem)
    clean = re.sub(r"\(\d+\)", " duplicate", stem)
    clean = re.sub(r"[^A-Za-z0-9]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean


def planned_entries(paths: list[Path]) -> list[Entry]:
    dated = [(p, parse_deck_date(p.stem)) for p in paths]
    dated.sort(key=lambda item: (item[1], item[0].name.lower()))

    per_day_counts: dict[date, int] = {}
    entries: list[Entry] = []
    for path, dt in dated:
        per_day_counts[dt] = per_day_counts.get(dt, 0) + 1
        idx = per_day_counts[dt]
        slug = readable_slug(path.stem)
        suffix = path.suffix.lower()
        new_name = f"{dt.isoformat()}_{idx:02d}_{slug}{suffix}"
        entries.append(Entry(path, path.name, dt, idx, new_name))
    return entries


def rename_entries(entries: list[Entry]) -> None:
    for entry in entries:
        tmp = entry.old_path.with_name(f".tmp_chrono_{entry.old_path.name}")
        if entry.old_path.name == entry.new_name:
            continue
        if tmp.exists():
            tmp.unlink()
        entry.old_path.rename(tmp)

    for entry in entries:
        tmp = entry.old_path.with_name(f".tmp_chrono_{entry.old_path.name}")
        new_path = entry.old_path.with_name(entry.new_name)
        if entry.old_path.name == entry.new_name:
            continue
        if new_path.exists():
            raise FileExistsError(new_path)
        tmp.rename(new_path)


def read_source_name(md_path: Path) -> str | None:
    for line in md_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:8]:
        match = re.search(r"Source PPTX:\s+`(.+?)`", line)
        if match:
            return match.group(1)
    return None


def build_markdown_entries(md_dir: Path) -> list[Entry]:
    paths = sorted(md_dir.glob("*.md"))
    synthetic: list[Path] = []
    mapping: dict[Path, Path] = {}

    for path in paths:
        source_name = read_source_name(path)
        parse_name = source_name or path.name
        fake_path = path.with_name(parse_name).with_suffix(path.suffix)
        synthetic.append(fake_path)
        mapping[fake_path] = path

    entries = planned_entries(synthetic)
    actual_entries: list[Entry] = []
    for entry in entries:
        actual = mapping[entry.old_path]
        actual_entries.append(Entry(actual, actual.name, entry.parsed_date, entry.per_day_index, entry.new_name))
    return actual_entries


def write_index(base: Path, md_entries: list[Entry], pptx_entries: list[Entry]) -> None:
    md_by_date_idx = {(e.parsed_date, e.per_day_index): e for e in md_entries}
    pptx_by_date_idx = {(e.parsed_date, e.per_day_index): e for e in pptx_entries}
    keys = sorted(set(md_by_date_idx) | set(pptx_by_date_idx))

    lines = [
        "# Weekly Meeting Decks: Chronological Index",
        "",
        "All files are named as:",
        "",
        "`YYYY-MM-DD_NN_original-name.ext`",
        "",
        "`NN` disambiguates multiple decks from the same date.",
        "",
        "| # | Date | Markdown | Original PPTX |",
        "|---:|---|---|---|",
    ]
    for idx, key in enumerate(keys, start=1):
        md = md_by_date_idx.get(key)
        pptx = pptx_by_date_idx.get(key)
        md_link = f"[{md.new_name}](<extracted text markdown/{md.new_name}>)" if md else ""
        pptx_text = f"`{pptx.new_name}`" if pptx else ""
        lines.append(f"| {idx} | {key[0].isoformat()} | {md_link} | {pptx_text} |")

    (base / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    base = Path(__file__).resolve().parent
    md_dir = base / "extracted text markdown"
    originals_dir = base / "originals"

    if not md_dir.exists():
        raise FileNotFoundError(md_dir)
    if not originals_dir.exists():
        raise FileNotFoundError(originals_dir)

    md_entries = build_markdown_entries(md_dir)
    pptx_entries = planned_entries(sorted(originals_dir.glob("*.pptx")))

    rename_entries(md_entries)
    rename_entries(pptx_entries)

    # Rebuild entries after renaming so index points to current filenames.
    md_entries = build_markdown_entries(md_dir)
    pptx_entries = planned_entries(sorted(originals_dir.glob("*.pptx")))
    write_index(base, md_entries, pptx_entries)

    print(f"Renamed {len(md_entries)} Markdown files in {md_dir}")
    print(f"Renamed {len(pptx_entries)} PPTX files in {originals_dir}")
    print(f"Updated {base / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
