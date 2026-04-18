"""Build a ground-visit → viewer-crown mapping for the standalone HTML viewers.

QField exports in `input/qfield ground visit annotated crowns/.../*.geojson` are actually
GeoPackages (GPKG) containing UTM geometries and (sometimes) a `chain_id` attribute.

The interactive viewers in:
- `output/*/interactive_*_viewer/index.html`

do NOT include `chain_id` in their embedded `MANIFEST`. Instead they index crowns by
`crown_index` (0..N-1).

We bridge ground-visit records to viewer crowns using the cleaned consensus crowns:
- `output/*/consensus_crowns_complete_all_*.gpkg`

The ground-visit crown IDs are not guaranteed to be stable across runs.
So the default matching method is geometric overlap (IoU) between QField polygons
and consensus polygons.

The viewer's `crown_index` numbering is assumed to correspond to the *row index*
in the consensus crowns file used to generate the viewer.

This script matches:

    consensus_row_index (== crown_index) <-> qfield_feature_index (by IoU)

and transfers QField attributes to the matched consensus crown.

and writes a small JS file that defines `window.GROUND_VISIT`.

Run (detectree conda env):

  /Users/hbot07/anaconda3/envs/detectree/bin/python \
    src/scripts/build_ground_visit_mapping.py \
    --qfield "input/qfield ground visit annotated crowns/LHC/output(Qfield)_Lhc(16-03-26).geojson" \
    --consensus "output/lhc_tracking_rerun_1Apr26/consensus_crowns_complete_all_lhc.gpkg" \
    --out "output/lhc_tracking_rerun_1Apr26/interactive_lhc_viewer/ground_visit.js"

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd


def _coalesce(row: Any, *keys: str) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _row_record_from_qfield(r: Any, qfield_feature_index: int) -> Dict[str, Any]:
    chain_id_val = _coalesce(r, "chain_id")
    rec: Dict[str, Any] = {
        "qfield_feature_index": qfield_feature_index,
        "qfield_chain_id": int(chain_id_val) if chain_id_val is not None else None,
        "species": _coalesce(r, "species", "Species"),
        "tree_type": _coalesce(r, "Tree type", "tree_type", "Tree_type"),
        "health_class": _coalesce(r, "health class", "health_class", "Health class", "Health_class"),
        "status": _coalesce(r, "status", "Status"),
        "description": _coalesce(r, "description", "Description"),
        "photo": _coalesce(r, "photo", "Photo"),
    }

    # Drop keys with None to keep payload small.
    rec = {k: v for k, v in rec.items() if v is not None}
    return rec


def _match_by_iou(
    qfield: gpd.GeoDataFrame,
    consensus: gpd.GeoDataFrame,
    min_iou: float,
) -> Tuple[Dict[int, int], List[int], List[int], Dict[str, Any]]:
    # Build candidate pairs using a spatial index, then do a greedy one-to-one assignment
    # by descending IoU.
    # Ensure positional indices align with viewer crown_index expectations.
    qfield = qfield.reset_index(drop=True)
    consensus = consensus.reset_index(drop=True)

    if qfield.crs is not None and consensus.crs is not None and qfield.crs != consensus.crs:
        qfield = qfield.to_crs(consensus.crs)

    sindex = consensus.sindex

    pairs: List[Tuple[float, float, float, int, int]] = []
    for q_idx, q_geom in enumerate(qfield.geometry):
        if q_geom is None or q_geom.is_empty:
            continue

        cand = list(sindex.intersection(q_geom.bounds))
        if not cand:
            continue

        c_geoms = consensus.geometry.iloc[cand]
        inter = c_geoms.intersection(q_geom)
        inter_area = inter.area.to_numpy()
        if (inter_area <= 0).all():
            continue

        c_area = c_geoms.area.to_numpy()
        q_area = float(q_geom.area)
        union_area = c_area + q_area - inter_area
        iou = inter_area / union_area
        ioq = inter_area / q_area if q_area else inter_area * 0.0
        ioc = inter_area / c_area

        for local_i, c_pos in enumerate(cand):
            v_iou = float(iou[local_i])
            if v_iou <= 0:
                continue
            pairs.append((v_iou, float(ioq[local_i]), float(ioc[local_i]), q_idx, int(c_pos)))

    pairs.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    assigned_q: set[int] = set()
    assigned_c: set[int] = set()
    q_to_c: Dict[int, int] = {}

    for v_iou, v_ioq, v_ioc, q_idx, c_idx in pairs:
        if v_iou < min_iou:
            break
        if q_idx in assigned_q or c_idx in assigned_c:
            continue
        assigned_q.add(q_idx)
        assigned_c.add(c_idx)
        q_to_c[q_idx] = c_idx

    unmatched_q = [i for i in range(len(qfield)) if i not in assigned_q]
    unmatched_c = [i for i in range(len(consensus)) if i not in assigned_c]

    debug = {
        "min_iou": min_iou,
        "pair_count": len(pairs),
        "assigned_count": len(q_to_c),
    }
    return q_to_c, unmatched_q, unmatched_c, debug


def _build_payload(qfield_path: Path, consensus_path: Path, *, match: str, min_iou: float) -> Dict[str, Any]:
    consensus = gpd.read_file(consensus_path)
    if "geometry" not in consensus.columns:
        raise ValueError(f"Consensus file missing geometry: {consensus_path}")

    qfield = gpd.read_file(qfield_path)
    if "geometry" not in qfield.columns:
        raise ValueError(f"QField file missing geometry: {qfield_path}")

    # Build canonical QField attribute rows (by feature index).
    qfield = qfield.reset_index(drop=True)
    consensus = consensus.reset_index(drop=True)

    qfield_df = qfield.drop(columns=[c for c in qfield.columns if c == "geometry"]).copy()
    qfield_records_by_feature_index: Dict[int, Dict[str, Any]] = {}
    for i, (_, r) in enumerate(qfield_df.iterrows()):
        qfield_records_by_feature_index[i] = _row_record_from_qfield(r, qfield_feature_index=i)

    if match == "iou":
        q_to_c, unmatched_q, unmatched_c, debug = _match_by_iou(qfield=qfield, consensus=consensus, min_iou=min_iou)

        by_crown_index: Dict[str, Dict[str, Any]] = {}
        for q_idx, c_idx in q_to_c.items():
            rec = dict(qfield_records_by_feature_index[q_idx])
            rec["match"] = {
                "type": "iou",
                "qfield_feature_index": q_idx,
                "consensus_row_index": c_idx,
            }

            # Recompute match metrics for the selected pair (cheap, one per match).
            qg = qfield.geometry.iloc[q_idx]
            cg = consensus.geometry.iloc[c_idx]
            inter_area = cg.intersection(qg).area
            union_area = cg.area + qg.area - inter_area
            iou = float(inter_area / union_area) if union_area else 0.0
            ioq = float(inter_area / qg.area) if qg.area else 0.0
            ioc = float(inter_area / cg.area) if cg.area else 0.0

            rec["match"]["confidence"] = iou
            rec["match"]["iou"] = iou
            rec["match"]["intersection_over_qfield"] = ioq
            rec["match"]["intersection_over_consensus"] = ioc

            # Add consensus chain_id for debugging if present.
            if "chain_id" in consensus.columns:
                rec["consensus_chain_id"] = int(consensus["chain_id"].iloc[c_idx])

            by_crown_index[str(c_idx)] = rec

        unmatched_q_chain_ids: List[int] = []
        if "chain_id" in qfield.columns:
            for q_idx in unmatched_q:
                try:
                    unmatched_q_chain_ids.append(int(qfield["chain_id"].iloc[q_idx]))
                except Exception:
                    continue

        return {
            "schema_version": 2,
            "qfield_source": str(qfield_path.as_posix()),
            "consensus_source": str(consensus_path.as_posix()),
            "match_method": "iou_greedy_one_to_one",
            "match_debug": debug,
            "matched_count": len(by_crown_index),
            "qfield_record_count": len(qfield),
            "consensus_crown_count": len(consensus),
            "unmatched_qfield_feature_indices": unmatched_q,
            "unmatched_qfield_chain_ids": unmatched_q_chain_ids,
            "unmatched_consensus_row_indices": unmatched_c,
            "by_crown_index": by_crown_index,
        }

    raise ValueError(f"Unknown match method: {match}")


def _write_output(payload: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".json":
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    # Default to JS for viewer embedding.
    js = "window.GROUND_VISIT = " + json.dumps(payload, ensure_ascii=False) + ";\n"
    out_path.write_text(js, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ground-visit mapping for interactive viewers")
    parser.add_argument("--qfield", required=True, help="Path to QField export (GPKG, may have .geojson extension)")
    parser.add_argument("--consensus", required=True, help="Path to consensus crowns GPKG")
    parser.add_argument("--out", required=True, help="Output path (.js or .json)")
    parser.add_argument(
        "--match",
        choices=["iou"],
        default="iou",
        help="Matching method (default: iou)",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.05,
        help="Minimum IoU required to accept a match (default: 0.05)",
    )

    args = parser.parse_args()

    qfield_path = Path(args.qfield)
    consensus_path = Path(args.consensus)
    out_path = Path(args.out)

    payload = _build_payload(
        qfield_path=qfield_path,
        consensus_path=consensus_path,
        match=args.match,
        min_iou=args.min_iou,
    )
    _write_output(payload, out_path)


if __name__ == "__main__":
    main()
