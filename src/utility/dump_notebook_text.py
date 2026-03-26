"""Dump a Jupyter .ipynb to a readable text file.

- Writes every cell's source.
- Writes all outputs except image/* mime types.

This is intended for safe inspection when notebooks contain large base64 images.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable


SKIP_MIME_PREFIXES = ("image/",)


def _as_text(source: Any) -> str:
    if isinstance(source, list) and all(isinstance(x, str) for x in source):
        return "".join(source)
    if source is None:
        return ""
    return str(source)


def _write_data_block(f, mime: str, data: Any) -> None:
    f.write(f"[mime] {mime}\n")

    if isinstance(data, str):
        f.write(data)
        if not data.endswith("\n"):
            f.write("\n")
        return

    if isinstance(data, list) and all(isinstance(x, str) for x in data):
        text = "".join(data)
        f.write(text)
        if text and not text.endswith("\n"):
            f.write("\n")
        return

    f.write(json.dumps(data, ensure_ascii=False, indent=2))
    f.write("\n")


def dump_notebook(ipynb_path: Path, out_path: Path) -> None:
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    cells: list[dict[str, Any]] = nb.get("cells", [])

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"NOTEBOOK: {ipynb_path.as_posix()}\n")
        f.write(f"N_CELLS: {len(cells)}\n\n")

        for idx, cell in enumerate(cells, start=1):
            ctype = cell.get("cell_type")
            exec_count = cell.get("execution_count")

            f.write("=" * 100 + "\n")
            f.write(f"CELL {idx} | type={ctype} | execution_count={exec_count}\n")
            f.write("-" * 100 + "\n")

            src_text = _as_text(cell.get("source", ""))
            f.write("[source]\n")
            f.write(src_text)
            if src_text and not src_text.endswith("\n"):
                f.write("\n")

            outputs: Iterable[dict[str, Any]] = cell.get("outputs", []) if ctype == "code" else []
            outputs = list(outputs)
            if outputs:
                f.write("\n[outputs]\n")

            for j, out in enumerate(outputs, start=1):
                otype = out.get("output_type")
                f.write("-" * 60 + "\n")
                f.write(f"output {j} | output_type={otype}\n")

                if otype == "stream":
                    f.write(f"[name] {out.get('name')}\n")
                    text = _as_text(out.get("text", ""))
                    f.write(text)
                    if text and not text.endswith("\n"):
                        f.write("\n")
                    continue

                if otype in ("execute_result", "display_data"):
                    data = out.get("data", {}) or {}
                    for mime in sorted(data.keys()):
                        if any(mime.startswith(p) for p in SKIP_MIME_PREFIXES):
                            f.write(f"[mime] {mime} (SKIPPED)\n")
                            continue
                        _write_data_block(f, mime, data[mime])
                    continue

                if otype == "error":
                    f.write(f"[ename] {out.get('ename')}\n")
                    f.write(f"[evalue] {out.get('evalue')}\n")
                    tb = out.get("traceback", [])
                    f.write("[traceback]\n")
                    if isinstance(tb, list) and all(isinstance(x, str) for x in tb):
                        f.write("\n".join(tb) + "\n")
                    else:
                        f.write(str(tb) + "\n")
                    continue

                f.write(json.dumps(out, ensure_ascii=False, indent=2))
                f.write("\n")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("Usage: dump_notebook_text.py <input.ipynb> <output.txt>", file=sys.stderr)
        return 2

    ipynb_path = Path(argv[1])
    out_path = Path(argv[2])

    dump_notebook(ipynb_path, out_path)
    print(f"WROTE {out_path} bytes={out_path.stat().st_size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
