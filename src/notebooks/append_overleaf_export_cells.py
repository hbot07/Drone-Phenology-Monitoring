#!/usr/bin/env python3

from __future__ import annotations

import json
import uuid
from pathlib import Path


MARKER = "Overleaf export (leaf-shed classifier figures)"


def _new_id() -> str:
    # Match the existing VS Code cell-id style.
    return "#VSC-" + uuid.uuid4().hex[:8]


def _has_marker(nb: dict) -> bool:
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source") or [])
        if MARKER in src:
            return True
    return False


def main() -> int:
    nb_path = Path("src/notebooks/Phenology_signals_10Mar26.ipynb")
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    if _has_marker(nb):
        print("Marker already present; no changes made.")
        return 0

    md_cell = {
        "cell_type": "markdown",
        "id": _new_id(),
        "metadata": {"language": "markdown"},
        "source": [
            f"# {MARKER}\n",
            "\n",
            "Regenerates the Overleaf-ready figures used by `leafshed_classifier.tex`.\n",
        ],
    }

    code_cell = {
        "cell_type": "code",
        "id": _new_id(),
        "metadata": {"language": "python"},
        "execution_count": None,
        "outputs": [],
        "source": [
            "from __future__ import annotations\n",
            "\n",
            "import subprocess\n",
            "import sys\n",
            "from pathlib import Path\n",
            "\n",
            "# Find repo root (expects output/ at the root)\n",
            "root = Path().resolve()\n",
            "for _ in range(6):\n",
            "    if (root / 'output').exists():\n",
            "        break\n",
            "    root = root.parent\n",
            "\n",
            "script = root / 'src' / 'notebooks' / 'generate_leafshed_overleaf_figs.py'\n",
            "print('Running:', script)\n",
            "subprocess.check_call([sys.executable, str(script)])\n",
            "print('Done.')\n",
        ],
    }

    nb.setdefault("cells", []).extend([md_cell, code_cell])

    nb_path.write_text(
        json.dumps(nb, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    print("Appended Overleaf export cells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
