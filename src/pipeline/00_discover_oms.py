#!/usr/bin/env python3
"""
Pipeline Step 0: Orthomosaic Discovery and Configuration

Scans an orthomosaic directory for .tif files, sorts them chronologically
using filename date parsing, applies optional exclusions, and writes a
pipeline_config.json that all subsequent pipeline steps read.

Usage:
    python 00_discover_oms.py \\
        --om-dir /path/to/orthomosaics \\
        --output-dir /path/to/output/my_run \\
        [--model-path /path/to/250312_flexi.pth] \\
        [--exclude-stems stem1,stem2] \\
        [--tile-width 25] [--tile-height 25] [--tile-buffer 15] \\
        [--run-name my_custom_run_name] \\
        [--print-config]

Writes:
    <output_dir>/pipeline_config.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def find_project_root(start: Optional[Path] = None) -> Path:
    p = (start or Path(__file__).parent).resolve()
    for _ in range(12):
        if (p / "output").exists() and (p / "src").exists() and (p / "input").exists():
            return p
        p = p.parent
    raise FileNotFoundError(
        "Cannot locate project root (expected output/, src/, input/ siblings)."
    )


# ---------------------------------------------------------------------------
# Date parsing for chronological sort
# ---------------------------------------------------------------------------

def _parse_legacy_lhc_date(stem: str) -> Optional[date]:
    """Parse legacy LHC filename pattern: odm_orthophoto{D}_{M}_{YY} or odm_orthophoto_{D}_{M}_{YY}."""
    m = re.search(r'odm_orthophoto_?(\d{1,2})_(\d{1,2})_(\d{2})$', stem)
    if m:
        try:
            return date(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


def _parse_stem_date(stem: str) -> Optional[date]:
    """Parse current repo stems like lhc_DD-MM-YY, sit_DD-MM-YY_dateNotConfirmed, sv_spotX_DD-MM-YY."""
    m = re.search(
        r'(?:^|_)(\d{1,2})-(\d{1,2})-(\d{2})(?:_(?:dateUnknown|dateNotConfirmed))?$',
        stem,
        re.IGNORECASE,
    )
    if m:
        try:
            return date(2000 + int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            return None
    return _parse_legacy_lhc_date(stem)


def _parse_sit_number(stem: str) -> Optional[int]:
    """Parse sequential number from SIT-style filename: sit_om{N}."""
    m = re.search(r'sit_om(\d+)$', stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def sort_key(stem: str) -> Tuple:
    """Returns a sort key: (priority_tier, value) so dates < numbers < alphabetical."""
    d = _parse_stem_date(stem)
    if d is not None:
        return (0, d.toordinal(), stem)
    n = _parse_sit_number(stem)
    if n is not None:
        return (1, n, stem)
    return (2, 0, stem)


def _candidate_crown_stems(stem: str) -> List[str]:
    """Return compatible crown filename stems for renamed placeholder-date rasters."""
    candidates = [stem]
    if "dateNotConfirmed" in stem:
        candidates.append(stem.replace("dateNotConfirmed", "dateUnknown"))
    elif "dateUnknown" in stem:
        candidates.append(stem.replace("dateUnknown", "dateNotConfirmed"))
    return candidates


def _canonical_placeholder_stem(stem: str) -> str:
    return stem.replace("dateUnknown", "dateNotConfirmed")


def _resolve_existing_crown_gpkg(existing_crowns_dir: Path, stem: str, stem_to_gpkg: dict) -> str:
    for candidate_stem in _candidate_crown_stems(stem):
        if candidate_stem in stem_to_gpkg:
            return stem_to_gpkg[candidate_stem]
        candidate_path = existing_crowns_dir / f"{candidate_stem}_multithreshold.gpkg"
        if candidate_path.exists():
            return str(candidate_path)
    return str(existing_crowns_dir / f"{stem}_multithreshold.gpkg")


def discover_and_sort_stems(om_dir: Path, exclude_stems: List[str]) -> List[str]:
    tif_files = sorted(om_dir.glob("*.tif")) + sorted(om_dir.glob("*.TIF"))
    # Remove exact duplicates and legacy/current placeholder-date duplicate names.
    # If both *_dateUnknown and *_dateNotConfirmed are present, keep the clearer
    # *_dateNotConfirmed file but still allow old crown GPKGs via compatibility lookup.
    seen = {}
    for f in tif_files:
        key = _canonical_placeholder_stem(f.stem)
        current = seen.get(key)
        if current is None or (
            "dateNotConfirmed" in f.stem and "dateNotConfirmed" not in current.stem
        ):
            seen[key] = f
    unique = list(seen.values())

    exclude_set = set(exclude_stems)
    filtered = [f for f in unique if f.stem not in exclude_set]

    if not filtered:
        raise FileNotFoundError(
            f"No .tif files found in {om_dir}"
            + (f" after excluding {exclude_stems}" if exclude_set else "")
        )

    sorted_stems = sorted([f.stem for f in filtered], key=sort_key)
    return sorted_stems


# ---------------------------------------------------------------------------
# Model path discovery
# ---------------------------------------------------------------------------

def find_model_path(project_root: Path, hint: Optional[str] = None) -> str:
    if hint:
        p = Path(hint)
        if not p.is_absolute():
            p = (project_root / hint).resolve()
        if p.exists():
            return str(p)
        raise FileNotFoundError(f"Model not found at: {p}")

    # Auto-discover
    candidates = sorted((project_root / "input" / "detectree_models").glob("*.pth"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(
        "No .pth model found under input/detectree_models/. "
        "Pass --model-path explicitly."
    )


# ---------------------------------------------------------------------------
# Config writer
# ---------------------------------------------------------------------------

def build_config(
    om_dir: Path,
    output_dir: Path,
    project_root: Path,
    stems: List[str],
    model_path: str,
    exclude_stems: List[str],
    tile_width: int,
    tile_height: int,
    tile_buffer: int,
    run_name: str,
) -> dict:
    pairs = [
        [
            str(output_dir / "01_detectree" / "crowns_multithreshold" / f"{s}_multithreshold.gpkg"),
            str(om_dir / f"{s}.tif"),
            s,
        ]
        for s in stems
    ]
    return {
        "run_name": run_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "project_root": str(project_root),
        "om_dir": str(om_dir),
        "output_dir": str(output_dir),
        "model_path": model_path,
        "om_stems": stems,
        "exclude_stems": exclude_stems,
        "pairs": pairs,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tile_buffer": tile_buffer,
        "detectree_dir": str(output_dir / "01_detectree"),
        "crowns_dir": str(output_dir / "01_detectree" / "crowns_multithreshold"),
        "tracking_dir": str(output_dir / "02_tracking"),
        "phenology_dir": str(output_dir / "03_phenology"),
        "viewer_dir": str(output_dir / "04_viewer"),
        "steps_completed": [],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover orthomosaics and write pipeline_config.json"
    )
    parser.add_argument("--om-dir", required=True, help="Folder containing .tif orthomosaics")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Pipeline output directory (default: <project_root>/output/<run_name>)",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to .pth model file (default: auto-discover under input/detectree_models/)",
    )
    parser.add_argument(
        "--exclude-stems",
        default="",
        help="Comma-separated list of stems to exclude from tracking (e.g. bad OMs)",
    )
    parser.add_argument("--tile-width", type=int, default=25)
    parser.add_argument("--tile-height", type=int, default=25)
    parser.add_argument("--tile-buffer", type=int, default=15)
    parser.add_argument(
        "--run-name",
        default=None,
        help="Human-readable run name (default: derived from om_dir folder name + timestamp)",
    )
    parser.add_argument(
        "--crowns-dir",
        default=None,
        help=(
            "Use an existing crowns directory instead of the default "
            "<output_dir>/01_detectree/crowns_multithreshold/. "
            "Useful when detectree detection was already run separately."
        ),
    )
    parser.add_argument(
        "--only-stems",
        default="",
        help="Comma-separated subset of stems to include (e.g. sit_om3,sit_om4,sit_om5 for quick test runs)",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the config JSON to stdout after writing",
    )
    args = parser.parse_args()

    om_dir = Path(args.om_dir).resolve()
    if not om_dir.exists():
        print(f"ERROR: orthomosaic directory not found: {om_dir}", file=sys.stderr)
        return 1

    project_root = find_project_root(om_dir)

    exclude_stems = [s.strip() for s in args.exclude_stems.split(",") if s.strip()]

    print(f"Scanning {om_dir} for orthomosaics...")
    if exclude_stems:
        print(f"  Excluding: {exclude_stems}")

    stems = discover_and_sort_stems(om_dir, exclude_stems)
    print(f"Found {len(stems)} orthomosaics (chronological order):")
    for i, s in enumerate(stems, 1):
        print(f"  OM{i:02d}: {s}")

    try:
        model_path = find_model_path(project_root, args.model_path)
        print(f"Model: {model_path}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Build run name
    if args.run_name:
        run_name = args.run_name
    else:
        folder_tag = om_dir.name.lower().replace("input_om_", "").replace("input_", "")
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_name = f"pipeline_{folder_tag}_{ts}"

    # Resolve output_dir
    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
    else:
        output_dir = (project_root / "output" / run_name).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)

    # Allow using an existing crowns directory (e.g. from a prior detectree run)
    existing_crowns_dir: Optional[str] = None
    if getattr(args, "crowns_dir", None):
        existing_crowns_dir = str(Path(args.crowns_dir).resolve())
        print(f"Using existing crowns dir: {existing_crowns_dir}")

    # Apply --only-stems filter for quick isolated test runs
    only_stems = [s.strip() for s in getattr(args, "only_stems", "").split(",") if s.strip()]
    if only_stems:
        missing = [s for s in only_stems if s not in stems]
        if missing:
            print(f"WARNING: --only-stems requested stems not found in om_dir: {missing}", file=sys.stderr)
        stems = [s for s in stems if s in only_stems]
        if not stems:
            print(f"ERROR: --only-stems filtered out all stems", file=sys.stderr)
            return 1
        print(f"  Filtered to {len(stems)} stems via --only-stems: {stems}")

    config = build_config(
        om_dir=om_dir,
        output_dir=output_dir,
        project_root=project_root,
        stems=stems,
        model_path=model_path,
        exclude_stems=exclude_stems,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        tile_buffer=args.tile_buffer,
        run_name=run_name,
    )

    # Override crowns_dir + pairs if an existing crowns dir was specified
    if existing_crowns_dir:
        config["crowns_dir"] = existing_crowns_dir
        # Load run_summary.json if present to resolve correct GPKG paths
        # (detection runs may tag files as OM1, OM2,... rather than stem names)
        stem_to_gpkg: dict = {}
        for summary_candidate in [
            Path(existing_crowns_dir) / "run_summary.json",
            Path(existing_crowns_dir).parent / "run_summary.json",
        ]:
            if summary_candidate.exists():
                import json as _json
                summary_data = _json.loads(summary_candidate.read_text())
                if isinstance(summary_data, list):
                    for entry in summary_data:
                        om_stem = entry.get("orthomosaic") or entry.get("stem")
                        gpkg_path = entry.get("gpkg")
                        if om_stem and gpkg_path and Path(gpkg_path).exists():
                            stem_to_gpkg[om_stem] = gpkg_path
                if stem_to_gpkg:
                    print(f"  Loaded {len(stem_to_gpkg)} GPKG paths from {summary_candidate.name}")
                break
        config["pairs"] = [
            [
                _resolve_existing_crown_gpkg(Path(existing_crowns_dir), s, stem_to_gpkg),
                str(om_dir / f"{s}.tif"),
                s,
            ]
            for s in stems
        ]

    config_path = output_dir / "pipeline_config.json"
    config_path.write_text(json.dumps(config, indent=2))
    print(f"\nConfig written to: {config_path}")

    if args.print_config:
        print(json.dumps(config, indent=2))

    # Always print the config path last so the shell script can capture it
    print(f"PIPELINE_CONFIG={config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
