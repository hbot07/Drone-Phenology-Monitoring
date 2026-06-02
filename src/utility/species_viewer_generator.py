#!/usr/bin/env python3
"""
Generate a species-filtered SIT crown tracking viewer.

This is intentionally separate from src/notebooks/pipeline. It reuses the
already-generated Step 4 viewer assets, but reads species labels from the
master GeoJSON.

Reads:
  input/master geojsons/sit_master.geojson
  output/sit_tracking_rerun_1Apr26/interactive_sit_viewer/manifest.json
  output/sit_tracking_rerun_1Apr26/interactive_sit_viewer/crowns_underlay_OM01_sit_om1_pixels.geojson

Writes:
  output/sit_tracking_rerun_1Apr26/interactive_sit_viewer/species_viewer.html
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_COUNTS = {
    "Maulsari": 13,
    "Ashok": 13,
    "Mandphali": 11,
    "Neem": 11,
    "Pilkhan": 8,
    "Buddha's Coconut": 5,
    "Peepal": 3,
    "Caribbean trumpet, Ashok": 3,
    "Kachnar/Karanj": 3,
    "Amaltas": 3,
    "Chamrod": 2,
    "Marodphali": 2,
    "Amaltas, Peepal": 2,
    "Franjipani": 2,
    "Mahneem": 2,
    "Pilkhan or Semal fig": 1,
    "Semal": 1,
    "Caribbean trumpet": 1,
    "Caribbean trumpet, Buddha's Coconut": 1,
    "Karanj": 1,
    "Caribbean Trumpet": 1,
    "Palm tree": 1,
    "Palash": 1,
    "Siris": 1,
    "Kachnar": 1,
    "Mandphali, Kachnar": 1,
    "Arjun": 1,
}

SPECIES_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea", "#0891b2",
    "#ea580c", "#4f46e5", "#be123c", "#0f766e", "#7c2d12", "#65a30d",
    "#a21caf", "#0369a1", "#b45309", "#15803d", "#c026d3", "#0d9488",
    "#b91c1c", "#4338ca", "#854d0e", "#047857", "#7e22ce", "#1d4ed8",
    "#9f1239", "#166534", "#92400e",
]


def project_root(start: Path) -> Path:
    cur = start.resolve()
    for candidate in [cur, *cur.parents]:
        if (candidate / "src").exists() and (candidate / "input").exists():
            return candidate
    raise FileNotFoundError("Could not find project root")


def first_present(*values: Any) -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return None


def species_from_properties(props: dict[str, Any]) -> str | None:
    species = first_present(
        props.get("species"),
        (props.get("field_data") or {}).get("species"),
        (props.get("properties") or {}).get("species"),
    )
    if species is None:
        return None
    species = str(species).strip()
    if not species or species.lower() in {"none", "null", "nan", "unknown"}:
        return None
    return species


def crown_index_from_properties(props: dict[str, Any]) -> int | None:
    ids = props.get("ids") or {}
    value = first_present(props.get("crown_index"), ids.get("crown_index"))
    if value is None:
        label = first_present(props.get("crown_label"), ids.get("crown_label"), ids.get("crown_id"))
        if isinstance(label, str) and label.startswith("crown_"):
            try:
                return int(label.rsplit("_", 1)[1]) - 1
            except ValueError:
                return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def load_master(master_path: Path) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]], Counter[str]]:
    with master_path.open(encoding="utf-8") as f:
        data = json.load(f)

    by_index: dict[int, dict[str, Any]] = {}
    by_label: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    for feat in data.get("features", []):
        props = feat.get("properties") or {}
        crown_index = crown_index_from_properties(props)
        if crown_index is None:
            continue
        ids = props.get("ids") or {}
        species = species_from_properties(props)
        if species:
            counts[species] += 1
        record = {
            "crown_index": crown_index,
            "crown_id": first_present(ids.get("crown_id"), f"crown_{crown_index + 1:04d}"),
            "crown_label": first_present(ids.get("crown_label"), f"crown_{crown_index + 1:04d}"),
            "chain_id": ids.get("chain_id"),
            "species": species,
            "tree_type": (props.get("field_data") or {}).get("tree_type"),
            "health_class": (props.get("field_data") or {}).get("health_class"),
            "status": (props.get("field_data") or {}).get("status"),
            "description": (props.get("field_data") or {}).get("description"),
        }
        by_index[crown_index] = record
        for label in {record["crown_id"], record["crown_label"], feat.get("id")}:
            if isinstance(label, str):
                by_label[label.split(":", 1)[-1]] = record
    return by_index, by_label, counts


def build_records(
    master_by_index: dict[int, dict[str, Any]],
    master_by_label: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    pixel_geojson: dict[str, Any],
) -> list[dict[str, Any]]:
    manifest_by_index = {
        int(entry["crown_index"]): entry
        for entry in manifest.get("crowns", [])
    }
    pixel_by_index = {
        int(feat["properties"]["crown_index"]): {
            "geometry": feat["geometry"],
            "crown_label": feat["properties"].get("crown_label"),
        }
        for feat in pixel_geojson.get("features", [])
        if feat.get("properties", {}).get("crown_index") is not None
    }

    records: list[dict[str, Any]] = []
    for crown_index in sorted(pixel_by_index):
        asset_label = pixel_by_index[crown_index].get("crown_label")
        manifest_entry = manifest_by_index.get(crown_index, {})
        asset_id = manifest_entry.get("crown_id")
        master = (
            master_by_label.get(asset_label or "")
            or master_by_label.get(asset_id or "")
        )
        entry = manifest_entry
        items = []
        for item in entry.get("items", []):
            items.append({
                "om_id": item.get("om_id"),
                "stem": item.get("stem"),
                "status": item.get("status"),
                "path": item.get("path"),
            })
        records.append({
            "crown_index": crown_index,
            "crown_id": master.get("crown_id") or entry.get("crown_id") or asset_label or f"crown_{crown_index:04d}",
            "crown_label": master.get("crown_label") or asset_label or entry.get("crown_id") or f"crown_{crown_index:04d}",
            "chain_id": master.get("chain_id"),
            "species": master.get("species"),
            "tree_type": master.get("tree_type"),
            "health_class": master.get("health_class"),
            "status": master.get("status"),
            "description": master.get("description"),
            "geometry": pixel_by_index[crown_index]["geometry"],
            "items": items,
        })
    return records


def validation_message(counts: Counter[str]) -> str:
    expected = Counter(EXPECTED_COUNTS)
    if counts == expected:
        return "Species counts match expected table."

    missing = expected - counts
    extra = counts - expected
    parts = ["WARNING: species counts do not match expected table."]
    if missing:
        parts.append("Missing/low: " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    if extra:
        parts.append("Extra/high: " + ", ".join(f"{k}={v}" for k, v in extra.items()))
    return "\n".join(parts)


def species_palette(species: list[str]) -> dict[str, str]:
    palette = {sp: SPECIES_COLORS[i % len(SPECIES_COLORS)] for i, sp in enumerate(species)}
    palette["_unknown"] = "#6b7280"
    return palette


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def build_html(records: list[dict[str, Any]], manifest: dict[str, Any], counts: Counter[str]) -> str:
    species_order = [sp for sp, _ in counts.most_common()]
    colors = species_palette(species_order)
    total = len(records)
    identified = sum(1 for row in records if row.get("species"))
    unknown = total - identified
    count_status = validation_message(counts)
    status_class = "ok" if counts == Counter(EXPECTED_COUNTS) else "warn"

    buttons = [
        '<button class="filter active" data-species="all">All <b>{}</b></button>'.format(total),
        '<button class="filter" data-species="_identified">Identified <b>{}</b></button>'.format(identified),
    ]
    for species in species_order:
        buttons.append(
            '<button class="filter" data-species="{}"><span style="background:{}"></span>{} <b>{}</b></button>'.format(
                html.escape(species, quote=True),
                colors[species],
                html.escape(species),
                counts[species],
            )
        )
    buttons.append('<button class="filter" data-species="_unknown">Unknown <b>{}</b></button>'.format(unknown))

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SIT Species Tracking Viewer</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body {{ height: 100%; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #172033; }}
    body {{ display: grid; grid-template-columns: minmax(260px, 340px) 1fr minmax(330px, 440px); background: #f4f7f2; }}
    #filters {{ overflow: auto; border-right: 1px solid #d5dccf; background: #fffdf8; padding: 14px; }}
    #filters h1 {{ margin: 0 0 4px; font-size: 18px; }}
    #filters .meta {{ margin: 0 0 12px; font-size: 12px; color: #60705c; }}
    .filter {{ width: 100%; min-height: 34px; margin: 0 0 6px; padding: 7px 9px; border: 1px solid #d6decf; background: #ffffff; border-radius: 6px; cursor: pointer; display: flex; align-items: center; gap: 8px; text-align: left; font-size: 13px; color: #172033; }}
    .filter span {{ width: 10px; height: 10px; border-radius: 999px; flex: 0 0 auto; }}
    .filter b {{ margin-left: auto; color: #53634f; }}
    .filter.active {{ border-color: #1f7a4d; box-shadow: inset 3px 0 0 #1f7a4d; background: #eef8f1; }}
    .data-status {{ margin: 0 0 12px; padding: 9px; border-radius: 6px; font-size: 12px; line-height: 1.35; white-space: pre-wrap; }}
    .data-status.ok {{ border: 1px solid #b7d7c0; background: #eef8f1; color: #166534; }}
    .data-status.warn {{ border: 1px solid #f0c36d; background: #fff7dc; color: #7a4b00; }}
    #map {{ min-width: 0; min-height: 100%; background: #dde5d7; }}
    #panel {{ border-left: 1px solid #d5dccf; background: #ffffff; overflow: auto; padding: 14px; }}
    #summary {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-bottom: 12px; }}
    .stat {{ border: 1px solid #d6decf; border-radius: 6px; padding: 9px; background: #fbfcf9; }}
    .stat small {{ display: block; color: #60705c; font-size: 11px; }}
    .stat strong {{ display: block; font-size: 20px; margin-top: 2px; }}
    #selectedTitle {{ margin: 8px 0 4px; font-size: 16px; }}
    #selectedMeta {{ margin: 0 0 10px; color: #60705c; font-size: 12px; line-height: 1.4; }}
    #bulkTitle {{ margin: 12px 0 8px; font-size: 13px; color: #374151; }}
    #bulkGrid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }}
    .crown-card {{ border: 1px solid #d6decf; border-radius: 6px; overflow: hidden; background: #fbfcf9; cursor: pointer; }}
    .crown-card.active {{ border-color: #1f7a4d; box-shadow: 0 0 0 2px rgba(31, 122, 77, 0.18); }}
    .crown-card img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; background: #e5e7eb; }}
    .crown-card div {{ padding: 6px; font-size: 11px; display: flex; justify-content: space-between; gap: 6px; }}
    #strip {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }}
    .crop {{ position: relative; border: 1px solid #d6decf; border-radius: 6px; overflow: hidden; background: #f3f4f6; }}
    .crop img {{ width: 100%; aspect-ratio: 1; object-fit: cover; display: block; }}
    .crop span {{ position: absolute; left: 5px; top: 5px; background: rgba(0, 0, 0, .62); color: #fff; border-radius: 4px; padding: 2px 5px; font-size: 10px; }}
    @media (max-width: 960px) {{
      body {{ grid-template-columns: 1fr; grid-template-rows: auto 55vh auto; overflow: auto; }}
      #filters {{ border-right: 0; border-bottom: 1px solid #d5dccf; max-height: 34vh; }}
      #panel {{ border-left: 0; border-top: 1px solid #d5dccf; }}
    }}
  </style>
</head>
<body>
  <aside id="filters">
    <h1>SIT Species</h1>
    <p class="meta">Click a species to show all matching crowns and their tracking crops.</p>
    <div class="data-status {status_class}">{html.escape(count_status)}</div>
    {"".join(buttons)}
  </aside>
  <main id="map"></main>
  <aside id="panel">
    <div id="summary">
      <div class="stat"><small>Visible crowns</small><strong id="visibleCount">{total}</strong></div>
      <div class="stat"><small>Identified</small><strong>{identified}</strong></div>
    </div>
    <h2 id="selectedTitle">Select a crown</h2>
    <p id="selectedMeta">The crop strip for the clicked crown will appear here.</p>
    <div id="strip"></div>
    <h3 id="bulkTitle">Current species set</h3>
    <div id="bulkGrid"></div>
  </aside>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <script>
    const RECORDS = {json.dumps(records, separators=(",", ":"))};
    const COLORS = {json.dumps(colors, separators=(",", ":"))};
    const BASE_IMAGE = {json.dumps(manifest.get("base_image"))};
    const IMG_W = {int(manifest.get("image_w", 0))};
    const IMG_H = {int(manifest.get("image_h", 0))};
    const map = L.map("map", {{ crs: L.CRS.Simple, minZoom: -4, maxZoom: 2, zoomSnap: 0.25, attributionControl: false }});
    const bounds = [[0, 0], [IMG_H, IMG_W]];
    L.imageOverlay(BASE_IMAGE, bounds).addTo(map);
    map.fitBounds(bounds);

    let activeSpecies = "all";
    let selectedLayer = null;
    let selectedIndex = null;
    const layers = [];

    function colorFor(row) {{ return COLORS[row.species || "_unknown"] || COLORS._unknown; }}
    function matches(row) {{
      if (activeSpecies === "all") return true;
      if (activeSpecies === "_identified") return Boolean(row.species);
      if (activeSpecies === "_unknown") return !row.species;
      return row.species === activeSpecies;
    }}
    function styleFor(row, selected) {{
      const color = colorFor(row);
      return {{ color, fillColor: color, weight: selected ? 4 : 1.5, opacity: selected ? 1 : 0.9, fillOpacity: selected ? 0.42 : 0.18 }};
    }}
    function dimStyle() {{ return {{ color: "#6b7280", fillColor: "#6b7280", weight: 1, opacity: 0.28, fillOpacity: 0.04 }}; }}

    RECORDS.forEach(row => {{
      const layer = L.geoJSON({{ type: "Feature", geometry: row.geometry, properties: row }}, {{
        style: () => styleFor(row, false),
        onEachFeature: (_feature, lyr) => {{
          lyr.on("click", event => {{
            L.DomEvent.stopPropagation(event);
            selectCrown(row.crown_index);
          }});
          lyr.on("mouseover", () => lyr.setStyle({{ weight: 3, fillOpacity: 0.35 }}));
          lyr.on("mouseout", () => lyr.setStyle(matches(row) ? styleFor(row, row.crown_index === selectedIndex) : dimStyle()));
        }}
      }}).addTo(map);
      layers.push({{ row, layer }});
    }});

    function renderStrip(row) {{
      const okItems = (row.items || []).filter(item => item.path && item.status !== "no_overlap_or_error");
      document.getElementById("strip").innerHTML = okItems.map(item => `
        <div class="crop"><span>OM${{String(item.om_id).padStart(2, "0")}}</span><img src="${{item.path}}" loading="lazy" alt="${{item.stem || ""}}"></div>
      `).join("");
    }}

    function selectCrown(crownIndex) {{
      selectedIndex = crownIndex;
      const row = RECORDS.find(item => item.crown_index === crownIndex);
      if (!row) return;
      if (selectedLayer) selectedLayer.setStyle(styleFor(selectedLayer._speciesRow, false));
      const hit = layers.find(item => item.row.crown_index === crownIndex);
      selectedLayer = hit ? hit.layer.getLayers()[0] : null;
      if (selectedLayer) {{
        selectedLayer._speciesRow = row;
        selectedLayer.setStyle(styleFor(row, true));
        selectedLayer.bringToFront();
      }}
      document.getElementById("selectedTitle").textContent = row.crown_label || row.crown_id;
      document.getElementById("selectedMeta").textContent = `${{row.species || "Unknown species"}} | chain ${{row.chain_id ?? "n/a"}} | crown index ${{row.crown_index}}`;
      renderStrip(row);
      document.querySelectorAll(".crown-card").forEach(card => card.classList.toggle("active", Number(card.dataset.index) === crownIndex));
    }}

    function renderBulkGrid(visible) {{
      document.getElementById("bulkGrid").innerHTML = visible.map(row => {{
        const first = (row.items || []).find(item => item.path && item.status !== "no_overlap_or_error");
        return `
          <button class="crown-card" data-index="${{row.crown_index}}">
            <img src="${{first ? first.path : ""}}" loading="lazy" alt="">
            <div><span>${{row.crown_label || row.crown_id}}</span><b>${{row.chain_id ?? ""}}</b></div>
          </button>
        `;
      }}).join("");
      document.querySelectorAll(".crown-card").forEach(card => {{
        card.addEventListener("click", () => selectCrown(Number(card.dataset.index)));
      }});
    }}

    function applyFilter() {{
      const visible = [];
      layers.forEach(({{ row, layer }}) => {{
        const match = matches(row);
        if (match) visible.push(row);
        layer.eachLayer(lyr => lyr.setStyle(match ? styleFor(row, row.crown_index === selectedIndex) : dimStyle()));
      }});
      document.getElementById("visibleCount").textContent = visible.length;
      renderBulkGrid(visible);
      if (visible.length) {{
        const fit = layers
          .filter(item => matches(item.row))
          .map(item => item.layer.getBounds())
          .reduce((acc, cur) => acc.extend(cur));
        map.fitBounds(fit, {{ padding: [35, 35] }});
      }}
    }}

    document.querySelectorAll(".filter").forEach(button => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll(".filter").forEach(item => item.classList.remove("active"));
        button.classList.add("active");
        activeSpecies = button.dataset.species;
        applyFilter();
      }});
    }});

    applyFilter();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate SIT species tracking viewer")
    root = project_root(Path(__file__))
    default_viewer = root / "output" / "sit_tracking_rerun_1Apr26" / "interactive_sit_viewer"
    parser.add_argument("--master-geojson", default=str(root / "input" / "master geojsons" / "sit_master.geojson"))
    parser.add_argument("--viewer-dir", default=str(default_viewer))
    parser.add_argument("--output", default=str(default_viewer / "species_viewer.html"))
    parser.add_argument("--base-image", default="base_underlay_OM01_sit_om1.png")
    parser.add_argument("--crowns-geojson", default="crowns_underlay_OM01_sit_om1_pixels.geojson")
    args = parser.parse_args()

    master_path = Path(args.master_geojson)
    viewer_dir = Path(args.viewer_dir)
    output_path = Path(args.output)
    manifest_path = viewer_dir / "manifest.json"

    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["base_image"] = args.base_image
    manifest["crowns_geojson"] = args.crowns_geojson
    pixel_geojson_path = viewer_dir / manifest["crowns_geojson"]
    with pixel_geojson_path.open(encoding="utf-8") as f:
        pixel_geojson = json.load(f)
    if not manifest.get("image_w") or not manifest.get("image_h"):
        image_w, image_h = png_size(viewer_dir / manifest["base_image"])
        manifest["image_w"] = image_w
        manifest["image_h"] = image_h

    master_by_index, master_by_label, counts = load_master(master_path)
    records = build_records(master_by_index, master_by_label, manifest, pixel_geojson)

    print(f"Master GeoJSON : {master_path}")
    print(f"Viewer assets  : {viewer_dir}")
    print(f"Output HTML    : {output_path}")
    print(f"Loaded crowns  : {len(records)}")
    print(f"With species   : {sum(counts.values())}")
    print(validation_message(counts))

    if counts:
        print("\nSpecies counts:")
        for species, count in counts.most_common():
            print(f"  {species}: {count}")

    output_path.write_text(build_html(records, manifest, counts), encoding="utf-8")
    print(f"\nWrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
