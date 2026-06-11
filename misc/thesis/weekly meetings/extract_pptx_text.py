#!/usr/bin/env python3
"""Extract readable Markdown text from weekly-meeting PPTX files.

This avoids depending on LibreOffice/PowerPoint. PPTX files are zip archives
containing XML slide files; this script extracts slide text, speaker notes text
when present, and writes one Markdown file per deck plus an index.
"""

from __future__ import annotations

import argparse
import html
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET


NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
}


@dataclass
class DeckSummary:
    source_name: str
    markdown_name: str
    slide_count: int
    nonempty_slide_count: int
    word_count: int


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def slide_number(name: str) -> int:
    match = re.search(r"slide(\d+)\.xml$", name)
    return int(match.group(1)) if match else 10**9


def safe_stem(name: str) -> str:
    stem = Path(name).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "", stem)
    stem = re.sub(r"\s+", "_", stem)
    return stem or "deck"


def extract_paragraphs(xml_bytes: bytes) -> list[str]:
    root = ET.fromstring(xml_bytes)
    paragraphs: list[str] = []

    for para in root.findall(".//a:p", NS):
        runs: list[str] = []
        for node in para.iter():
            tag = node.tag.rsplit("}", 1)[-1]
            if tag == "t" and node.text:
                runs.append(node.text)
            elif tag == "br":
                runs.append("\n")

        text = "".join(runs)
        text = html.unescape(text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if text:
            paragraphs.append(text)

    # Remove immediate duplicates produced by repeated placeholders.
    deduped: list[str] = []
    for text in paragraphs:
        if not deduped or deduped[-1] != text:
            deduped.append(text)
    return deduped


def extract_deck(pptx_path: Path) -> tuple[list[list[str]], dict[int, list[str]]]:
    slides: list[list[str]] = []
    notes: dict[int, list[str]] = {}

    with zipfile.ZipFile(pptx_path) as zf:
        slide_names = sorted(
            (name for name in zf.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", name)),
            key=slide_number,
        )
        for name in slide_names:
            slides.append(extract_paragraphs(zf.read(name)))

        note_names = sorted(
            (name for name in zf.namelist() if re.match(r"ppt/notesSlides/notesSlide\d+\.xml$", name)),
            key=slide_number,
        )
        for name in note_names:
            num = slide_number(name)
            text = extract_paragraphs(zf.read(name))
            if text:
                notes[num] = text

    return slides, notes


def write_markdown(pptx_path: Path, md_path: Path) -> DeckSummary:
    slides, notes = extract_deck(pptx_path)
    lines: list[str] = []
    lines.append(f"# {pptx_path.stem}")
    lines.append("")
    lines.append(f"Source PPTX: `{pptx_path.name}`")
    lines.append("")
    lines.append(f"Slides: {len(slides)}")
    lines.append("")

    word_count = 0
    nonempty = 0
    for idx, paragraphs in enumerate(slides, start=1):
        if paragraphs:
            nonempty += 1
        lines.append(f"## Slide {idx}")
        lines.append("")
        if paragraphs:
            for para in paragraphs:
                word_count += len(re.findall(r"\w+", para))
                if "\n" in para:
                    lines.append(para)
                else:
                    lines.append(f"- {para}")
                lines.append("")
        else:
            lines.append("_No extracted text._")
            lines.append("")

        note_text = notes.get(idx)
        if note_text:
            lines.append("### Speaker Notes")
            lines.append("")
            for para in note_text:
                word_count += len(re.findall(r"\w+", para))
                lines.append(f"- {para}")
                lines.append("")

    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return DeckSummary(
        source_name=pptx_path.name,
        markdown_name=md_path.name,
        slide_count=len(slides),
        nonempty_slide_count=nonempty,
        word_count=word_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--copy-originals", action="store_true")
    args = parser.parse_args()

    src = args.src.expanduser().resolve()
    out = args.out.resolve()
    md_dir = out / "markdown"
    originals_dir = out / "originals"
    md_dir.mkdir(parents=True, exist_ok=True)
    if args.copy_originals:
        originals_dir.mkdir(parents=True, exist_ok=True)

    pptx_files = sorted(src.glob("*.pptx"), key=natural_key)
    summaries: list[DeckSummary] = []

    for pptx in pptx_files:
        md_name = safe_stem(pptx.name) + ".md"
        md_path = md_dir / md_name
        summary = write_markdown(pptx, md_path)
        summaries.append(summary)

        if args.copy_originals:
            shutil.copy2(pptx, originals_dir / pptx.name)

    index_lines = [
        "# Weekly Meeting Deck Text Index",
        "",
        f"Source folder: `{src}`",
        "",
        f"Decks processed: {len(summaries)}",
        "",
        "| # | Source PPTX | Markdown | Slides | Nonempty slides | Words |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for idx, item in enumerate(summaries, start=1):
        index_lines.append(
            f"| {idx} | `{item.source_name}` | `markdown/{item.markdown_name}` | "
            f"{item.slide_count} | {item.nonempty_slide_count} | {item.word_count} |"
        )

    (out / "index.md").write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    print(f"Processed {len(summaries)} decks into {md_dir}")
    print(f"Index: {out / 'index.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
