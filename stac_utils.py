"""
stac_utils.py — Build the STAC Item that the DPM export callback returns.

WHY THIS FILE EXISTS
--------------------
CoRE Stack cluster-docker-services §9 ("Always return output in STAC format")
requires every compute service to return its deliverable as a STAC 1.1.0 Item —
not a bare file path or ad-hoc JSON. This module builds that Item for the DPM
phenology vector (tree_master_geojson_phenoclf.geojson) and the response
envelope STACD reads.

AUTHORITATIVE SOURCES (validated 2026-09-26):
  - §9 reference example (NREGA vector layer) — every required key + shape:
    https://docs.core-stack.org/server/cluster-docker-services/#9-always-return-output-in-stac-format
  - Table extension v1.2.0 schema (table:columns / table:row_count / table:primary_geometry):
    https://stac-extensions.github.io/table/v1.2.0/schema.json
  - STACD §12 (Algorithm Response Handling) — the HTTP status contract the
    callback must honour (200 success / 400 skip / 404 skip / 500 fail):
    github.com/SaharshLaud/STACD_framework  README §12

WHAT CHANGED vs the earlier draft
---------------------------------
  + added top-level "collection"            (§9 required, was missing)
  + added per-link "title"                  (§9 required, was missing)
  + added assets.style (QGIS .qml)          (§9: provide when producible)
  + added assets.thumbnail (PNG)            (§9: provide when producible)
  + table:columns now uses the REAL pipeline output columns
    (from 03_phenology_analysis.py + 12_apply_phenophase_to_geojson.py:
     ids.chain_id, ids.crown_id, classification.is_deciduous /
     deciduous_score / leaf_off_start_om / full_leaf_off_om /
     leaf_on_return_om, per-observation phenophase + phenophase_source)
  + added table:row_count and table:primary_geometry (table ext v1.2.0)
  + build_response_envelope() returns the {status, asset_id, stac_items:[...]}
    shape STACD expects from an API-mode algorithm (see Custom LULC)
  + STAC Item id = run_id (unique by construction; directly traceable to the
    DPM run in the database — no need for a name-derived string)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# The exact schema URL §9 mandates for vector layers. Do not change the version
# without updating table:columns handling to match that schema.
TABLE_EXT = "https://stac-extensions.github.io/table/v1.2.0/schema.json"

# QGIS style repo CoRE Stack points at in §9. If/when a DPM phenology .qml is
# committed there, set its raw URL here; until then the style asset is omitted.
QGIS_STYLE_URL: Optional[str] = None  # e.g. "https://raw.githubusercontent.com/core-stack-org/QGIS-Styles/main/DronePhenology/phenophase.qml"


# ---------------------------------------------------------------------------
# table:columns — the schema of the phenology vector.
#
# The DPM GeoJSON nests its attributes (properties.ids.*, properties.classification.*,
# properties.observations[]). When the layer is published to GeoServer/WFS the
# queryable columns are the flattened significant fields below. List EVERY field
# a consumer would filter or style on (§9: "list every output field").
# ---------------------------------------------------------------------------
PHENOLOGY_TABLE_COLUMNS = [
    {"name": "geometry",          "type": "geometry", "description": "Tree-crown polygon (consensus medoid geometry across the tracked series)"},
    {"name": "chain_id",          "type": "int64",    "description": "Stable id of the tracked crown across all orthomosaic dates"},
    {"name": "crown_id",          "type": "object",   "description": "Human-readable crown label"},
    {"name": "crown_index",       "type": "int64",    "description": "Positional index of the crown within the run"},
    {"name": "is_deciduous",      "type": "bool",     "description": "Whether the crown was classified deciduous over the series"},
    {"name": "deciduous_score",   "type": "float64",  "description": "Continuous deciduousness score (0..1) from phenology analysis"},
    {"name": "leaf_off_start_om", "type": "int64",    "description": "Orthomosaic index at which leaf-off begins (null if none observed)"},
    {"name": "full_leaf_off_om",  "type": "int64",    "description": "Orthomosaic index at which the crown is fully leaf-off (null if none)"},
    {"name": "leaf_on_return_om", "type": "int64",    "description": "Orthomosaic index at which leaf-on returns (null if none observed)"},
    {"name": "chain_length",      "type": "int64",    "description": "Number of dates the crown was successfully tracked across"},
    {"name": "quality",           "type": "object",   "description": "Tracking quality label for the crown chain"},
    {"name": "phenophase",        "type": "object",   "description": "Per-observation phenophase class (leaf-on / leaf-off / transition) — smoothed prediction written by 12_apply_phenophase_to_geojson.py"},
    {"name": "phenophase_source", "type": "object",   "description": "Source of the phenophase label: 'gb_classifier_smoothed' (deciduous crowns) or 'evergreen_assigned' (evergreen crowns)"},
]


def _bbox_from_geojson(geojson: dict) -> list[float]:
    """Compute [west, south, east, north] over all features' coordinates."""
    xs: list[float] = []
    ys: list[float] = []

    def _walk(coords):
        if not coords:
            return
        if isinstance(coords[0], (int, float)):
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
            return
        for c in coords:
            _walk(c)

    for feat in geojson.get("features", []):
        geom = feat.get("geometry") or {}
        _walk(geom.get("coordinates"))

    if not xs or not ys:
        raise ValueError("Cannot compute bbox: no coordinates found in GeoJSON")
    return [min(xs), min(ys), max(xs), max(ys)]


def _bbox_to_polygon(bbox: list[float]) -> dict:
    """Turn a bbox into a closed GeoJSON Polygon (the §9 example uses the extent)."""
    w, s, e, n = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


def build_phenology_stac_item(
    *,
    run_id: str,
    geojson: dict,
    data_href: str,
    run_name: str = "",
    start_datetime: Optional[str] = None,
    end_datetime: Optional[str] = None,
    thumbnail_href: Optional[str] = None,
    collection: Optional[str] = None,
    params: Optional[dict] = None,
) -> dict:
    """
    Build a STAC 1.1.0 Item for the DPM phenology vector.

    Required inputs:
      run_id     — DPM run id. Used directly as the STAC Item id — unique by
                   construction and directly traceable to the database row.
      geojson    — the loaded tree_master_geojson_phenoclf.geojson (for bbox + row_count)
      data_href  — FETCHABLE url of the published layer (GeoServer WFS / S3 / export).
                   §9: this must NOT be a local file path.

    Optional:
      run_name           — human-readable run name (used in STAC title).
      start/end_datetime — ISO 8601 temporal coverage of the OM series.
      thumbnail_href     — PNG preview url if the pipeline produced one.
      collection         — parent collection id; defaults to "drone_phenology_monitoring".
      params             — the pipeline params used for this run. When given, they
                           are recorded as dpm:* processing-lineage properties
                           (DS weights, ds_threshold, detection/tracking settings)
                           so the STAC Item captures exactly how it was produced.
    """
    bbox = _bbox_from_geojson(geojson)
    geometry = _bbox_to_polygon(bbox)
    n_features = len(geojson.get("features", []))

    coll = collection or "drone_phenology_monitoring"
    item_id = run_id  # unique by construction; directly traceable to db.runs row

    now_iso = datetime.now(timezone.utc).isoformat()
    # Use provided datetimes, or fall back to "now" as a point-in-time
    start_dt = start_datetime
    end_dt = end_datetime

    title = f"Drone Phenology — {run_name}" if run_name else f"Drone Phenology — {run_id}"

    properties = {
        "title": title,
        "description": (
            "Per-crown vegetation phenophase vector produced by the Drone Phenology "
            "Monitoring pipeline. Each polygon is a tree crown tracked across a series "
            "of drone orthomosaics; attributes record deciduousness and leaf-off / "
            "leaf-on timing (as orthomosaic indices) plus per-observation phenophase. "
            "Source: drone orthomosaic series; method: Detectree2 crown detection, "
            "graph-based cross-date tracking, phenophase classification."
        ),
        "start_datetime": start_dt,
        "end_datetime": end_dt,
        "datetime": now_iso if (not start_dt and not end_dt) else None,
        "keywords": ["phenology", "drone", "tree crowns", "phenophase", "deciduous"],
        "table:columns": PHENOLOGY_TABLE_COLUMNS,
        "table:primary_geometry": "geometry",
        "table:row_count": n_features,
        # DPM lineage (namespaced, non-standard — safe extra properties)
        "dpm:run_id": run_id,
        "dpm:run_name": run_name,
    }

    # Processing lineage: record the pipeline params that produced this layer as
    # dpm:* properties (safe extra keys). Only keys actually supplied are written.
    if params:
        _LINEAGE_KEYS = (
            "model_type", "tile_width", "tile_height", "tile_buffer", "fixed_iou",
            "base_threshold_tag", "align_threshold_tag", "align_method",
            "w_veg_amp", "w_depth", "w_gcc_amp", "w_tex", "ds_threshold",
            "underlay_om", "exclude_stems",
        )
        for _k in _LINEAGE_KEYS:
            _v = params.get(_k)
            if _v is not None:
                properties[f"dpm:{_k}"] = _v

    assets = {
        "data": {
            "href": data_href,
            "type": "application/geo+json",
            "title": "Phenology Vector Layer",
            "roles": ["data"],
        }
    }
    if QGIS_STYLE_URL:
        assets["style"] = {
            "href": QGIS_STYLE_URL,
            "type": "application/xml",
            "title": "QGIS Style file",
            "roles": ["metadata"],
        }
    if thumbnail_href:
        assets["thumbnail"] = {
            "href": thumbnail_href,
            "type": "image/png",
            "title": "Thumbnail",
            "roles": ["thumbnail"],
        }

    links = [
        {"rel": "root",       "href": "../../../../../catalog.json", "type": "application/json", "title": "CoRE Stack Spatio Temporal Asset Catalog"},
        {"rel": "collection", "href": "../collection.json",          "type": "application/json", "title": coll},
        {"rel": "parent",     "href": "../collection.json",          "type": "application/json", "title": coll},
    ]

    return {
        "type": "Feature",
        "stac_version": "1.1.0",
        "stac_extensions": [TABLE_EXT],
        "id": item_id,
        "geometry": geometry,
        "bbox": bbox,
        "properties": properties,
        "links": links,
        "assets": assets,
        "collection": coll,
    }


def build_response_envelope(stac_item: dict, asset_id: Optional[str] = None) -> dict:
    """
    Wrap the STAC Item in the {status, asset_id, stac_items:[...]} envelope an
    API-mode STACD algorithm returns (Custom LULC pattern). Return this from
    POST /api/export-phenology with HTTP 200.
    """
    return {
        "status": "success",
        "asset_id": [asset_id or stac_item["id"]],
        "version": "1",
        "hosting_platform": "GeoServer",
        "stac_items": [stac_item],
    }


# ---------------------------------------------------------------------------
# STACD §12 — HTTP status contract for the callback (documented here so the
# endpoint in server.py stays consistent with what STACD does downstream):
#
#   200  success  -> asset registered as a DatasetInstance + STAC catalog item
#   400  skipped  -> invalid params; return {"error","message"}; no asset
#   404  skipped  -> no data for this location/params; {"error","message"}
#   500  failed   -> pipeline/computation error; {"error","message"}; DAG fails
#
# So /api/export-phenology should:
#   - raise/return 400 when required conf keys are missing/unparseable
#   - return 404 when the run produced no phenology GeoJSON (empty result)
#   - return 500 when the pipeline raised
#   - return 200 + build_response_envelope(...) on success
# ---------------------------------------------------------------------------
def build_error_response(error: str, message: str) -> dict:
    """Body for 400/404/500 responses (STACD reads error+message into logs)."""
    return {"error": error, "message": message}


if __name__ == "__main__":
    # Tiny self-check with a 2-feature fake layer.
    demo = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "Polygon",
             "coordinates": [[[77.16, 28.53], [77.20, 28.53], [77.20, 28.57], [77.16, 28.57], [77.16, 28.53]]]},
             "properties": {}},
            {"type": "Feature", "geometry": {"type": "Point", "coordinates": [77.18, 28.55]},
             "properties": {}},
        ],
    }
    item = build_phenology_stac_item(
        run_id="demo-run-123",
        run_name="Devprayag 2024",
        geojson=demo,
        data_href="https://geoserver.core-stack.org:8443/geoserver/dpm/ows?service=WFS&request=GetFeature&typeName=dpm:demo-run-123&outputFormat=application/json",
        start_datetime="2024-02-15T00:00:00Z",
        end_datetime="2024-11-20T00:00:00Z",
    )
    print(json.dumps(build_response_envelope(item), indent=2))
