#!/usr/bin/env python3
"""
Pipeline Step 4b: Interactive Crown Phenology Viewer

Reads the tile manifest from Step 4a (COG + XYZ tiles) and consensus crowns,
then generates a standalone HTML viewer (Leaflet-based) with:

  - XYZ tile layers for all OMs — browser fetches only tiles in viewport
  - Embedded tile map per crown in right panel — zoom to crown on click
  - Crown polygons coloured by phenology class (deciduous / evergreen / uncertain)
  - Pinned info panel: phenology score, GCC/veg amplitude, leaf-on/off with dates
  - Timeline slider — switch OMs; updates both main map and crown tile view
  - Unified search: by crown ID or species annotation
  - Filter pills: All / Deciduous / Evergreen / Uncertain
  - Dark / light theme toggle (SVG moon / sun icon)
  - Crown comparison panel — two slots, each with independent tile map + slider
  - Species annotation stored in browser localStorage
  - Draw tool for multi-crown region selection

Requires: dpm-tracking conda environment
    rasterio, geopandas, numpy, pandas

Depends on: 04a_cog_tiling.py (must run first)

Usage:
    python 04b_interactive_viz.py --config /path/to/pipeline_config.json
        [--skip-if-done]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Config + shared helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_config(config: dict, config_path: Path) -> None:
    config_path.write_text(json.dumps(config, indent=2))


def setup_app_dir(project_root: Path) -> None:
    app_dir = str(project_root / "src" / "flask_app_tracking")
    if app_dir not in sys.path:
        sys.path.insert(0, app_dir)


def build_pairs_and_om_stems(config: dict) -> Tuple[List[Tuple[str, str, str]], Dict[int, str]]:
    crowns_dir = Path(config["crowns_dir"])
    om_dir     = Path(config["om_dir"])
    pairs, om_stems = [], {}
    for i, (gpkg_raw, tif_raw, stem) in enumerate(config["pairs"], 1):
        gpkg_from_config = Path(gpkg_raw)
        tif_from_config  = Path(tif_raw)
        gpkg = str(gpkg_from_config) if gpkg_from_config.exists() else str(crowns_dir / f"{stem}_multithreshold.gpkg")
        tif  = str(tif_from_config)  if tif_from_config.exists()  else str(om_dir     / f"{stem}.tif")
        pairs.append((gpkg, tif, stem))
        om_stems[i] = stem
    return pairs, om_stems


def _largest_polygon(geom):
    from shapely.geometry import Polygon, MultiPolygon
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        parts = list(geom.geoms)
        return max(parts, key=lambda p: p.area) if parts else None
    return None


def load_phenology_data(phenology_dir: Path) -> Dict[str, Any]:
    """Load Step 03 scores CSV and return per-crown dicts keyed by crown_NNNN."""
    try:
        import pandas as pd
    except ImportError:
        print("  Warning: pandas not available — skipping phenology data")
        return {}

    scores_csv = phenology_dir / "leafshed_tree_scores.csv"
    pheno: Dict[str, Any] = {}

    if not scores_csv.exists():
        print(f"  Warning: {scores_csv.name} not found — run step 03 first")
        return pheno

    try:
        df = pd.read_csv(scores_csv)
        for i, row in df.iterrows():
            cid = f"crown_{i:04d}"
            pheno[cid] = {}
            is_deciduous = row.get("is_deciduous", False)
            pheno[cid]["phenology_class"] = "deciduous" if is_deciduous else "evergreen"
            raw_score = row.get("deciduous_score", None)
            try:
                pheno[cid]["deciduous_score"] = round(float(raw_score) * 100, 1) if raw_score is not None else None
            except (TypeError, ValueError):
                pheno[cid]["deciduous_score"] = None
            leaf_on_om  = row.get("leaf_on_return_om",  None)
            leaf_off_om = row.get("full_leaf_off_om",   None)
            pheno[cid]["leaf_on_om"]    = int(leaf_on_om)  if pd.notna(leaf_on_om)  else None
            pheno[cid]["leaf_off_om"]   = int(leaf_off_om) if pd.notna(leaf_off_om) else None
            pheno[cid]["gcc_amplitude"] = row.get("s_gcc_amp", None)
            pheno[cid]["veg_amplitude"] = row.get("s_veg_amp", None)
    except Exception as e:
        print(f"  Warning: could not read scores CSV: {e}")
        import traceback; traceback.print_exc()

    print(f"  Loaded phenology data for {len(pheno)} crowns")
    return pheno


def generate_phenology_overview(scores_csv, phases_csv, om_stems, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd
        scores_df = pd.read_csv(scores_csv)
        scores_df["deciduous_score"] = pd.to_numeric(scores_df.get("deciduous_score"), errors="coerce")
        scores_df = scores_df[np.isfinite(scores_df["deciduous_score"])].copy()
        threshold = 0.85
        n_decid = int((scores_df["deciduous_score"] >= threshold).sum())
        n_total = len(scores_df)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(scores_df["deciduous_score"].values, bins=30, color="#4e79a7", alpha=0.85)
        axes[0].axvline(threshold, color="#d62728", lw=2, alpha=0.9, label=f"thr={threshold}")
        axes[0].set_title(f"Deciduous score | n={n_total} | deciduous={n_decid} ({n_decid/max(n_total,1):.1%})")
        axes[0].set_xlabel("Deciduous score"); axes[0].set_ylabel("# crowns"); axes[0].legend()
        xcol = "A_veg" if "A_veg" in scores_df.columns else None
        ycol = "A_gcc" if "A_gcc" in scores_df.columns else None
        if xcol and ycol:
            is_decid = scores_df["deciduous_score"] >= threshold
            axes[1].scatter(
                pd.to_numeric(scores_df[xcol], errors="coerce"),
                pd.to_numeric(scores_df[ycol], errors="coerce"),
                s=10, alpha=0.5,
                c=is_decid.map({True: "#d62728", False: "#2ca02c"}),
            )
            axes[1].set_xlabel(xcol); axes[1].set_ylabel(ycol)
            axes[1].set_title("Amplitude scatter (red=deciduous, green=evergreen)")
        else:
            axes[1].axis("off")
        fig.tight_layout()
        plt.savefig(str(out_png), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote phenology overview: {out_png}")
    except Exception as e:
        print(f"Warning: could not generate phenology overview: {e}")


def build_html(
    dataset_name: str,
    run_tag: str,
    geojson_path_name: str,
    om_tile_manifest: str,   # JSON: [{om_id, stem, tile_url, bounds, min_zoom, max_zoom}, ...]
    underlay_om_id: int,
    geojson_inline: str,
    manifest_inline: str,
    pheno_inline: str,
    crs_epsg: int,           # source CRS for pixel↔latlon; 3857 after COG
) -> str:

    return f'''<!doctype html>
<html lang="en" data-theme="dark">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Interactive Visualizer</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY=" crossorigin=""/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.css"/>
  <style>
    :root {{
      --bg0:#13151b; --bg1:#1a1d25; --bg2:#20242e; --bg3:#272b38;
      --bd:#2c303e; --t0:#d6dae6; --t1:#8a93a8; --t2:#525a6e;
      --green:#6fa882; --amber:#b89550; --sage:#5f9e6a;
      --shadow:0 4px 20px rgba(0,0,0,.5); --map-bg:#181b22;
    }}
    [data-theme="light"] {{
      --bg0:#eef0f4; --bg1:#f7f8fa; --bg2:#ffffff; --bg3:#e6e9ef;
      --bd:#cdd1db; --t0:#1a1e28; --t1:#4a5268; --t2:#8a92a6;
      --green:#3d7a56; --amber:#8a6820; --sage:#3a8050;
      --shadow:0 4px 20px rgba(0,0,0,.12); --map-bg:#c8ccd8;
    }}
    *, *::before, *::after {{ box-sizing:border-box; margin:0; padding:0; }}
    html, body {{
      height:100%; font-family:'Segoe UI',system-ui,-apple-system,sans-serif;
      background:var(--bg0); color:var(--t0); overflow:hidden; font-size:13px;
      transition:background .25s,color .25s;
    }}

    /* TOP BAR */
    #topbar {{
      position:fixed; top:0; left:0; right:0; height:48px; z-index:2000;
      background:var(--bg1); border-bottom:1px solid var(--bd);
      display:flex; align-items:center; gap:10px; padding:0 14px;
      box-shadow:0 1px 6px rgba(0,0,0,.25);
    }}
    .logo {{ font-weight:600; font-size:14px; color:var(--t0); white-space:nowrap; }}
    .logo em {{ color:var(--green); font-style:normal; }}
    .divider {{ width:1px; height:22px; background:var(--bd); flex-shrink:0; }}
    #search-wrap {{ display:flex; }}
    #search-input {{
      width:200px; padding:5px 10px; border-radius:5px 0 0 5px;
      background:var(--bg2); border:1px solid var(--bd); border-right:none;
      color:var(--t0); font-size:12px; outline:none; transition:border-color .15s;
    }}
    #search-input:focus {{ border-color:var(--green); }}
    #search-btn {{
      padding:5px 11px; border-radius:0 5px 5px 0;
      border:1px solid var(--bd); border-left:none;
      background:var(--bg3); color:var(--t1); cursor:pointer; font-size:12px;
    }}
    #search-btn:hover {{ color:var(--t0); }}
    #filter-wrap {{ display:flex; gap:4px; align-items:center; }}
    .flabel {{ font-size:11px; color:var(--t2); margin-right:2px; }}
    .pill {{
      padding:3px 10px; border-radius:4px; font-size:11px; cursor:pointer;
      border:1px solid var(--bd); color:var(--t2); background:var(--bg1);
      user-select:none; transition:all .15s;
    }}
    .pill:hover {{ color:var(--t1); border-color:var(--t2); }}
    .pill.active[data-filter="all"]       {{ color:#8ab0c8; border-color:#4a6080; background:var(--bg3); }}
    .pill.active[data-filter="deciduous"] {{ color:var(--amber); border-color:var(--amber); background:var(--bg3); }}
    .pill.active[data-filter="evergreen"] {{ color:var(--sage);  border-color:var(--sage);  background:var(--bg3); }}
    .pill.active[data-filter="uncertain"] {{ color:var(--t1);    border-color:var(--t1);    background:var(--bg3); }}
    #crown-count-badge {{
      position:absolute; left:50%; transform:translateX(-50%);
      background:var(--bg2); border:1px solid var(--bd);
      border-radius:4px; padding:3px 14px;
      font-size:12px; color:var(--green); font-weight:600; white-space:nowrap;
    }}
    .topbar-btn {{
      padding:5px 11px; border-radius:5px; border:1px solid var(--bd);
      background:var(--bg2); color:var(--t1); cursor:pointer; font-size:12px;
      transition:all .15s; display:flex; align-items:center; gap:5px;
    }}
    .topbar-btn:hover {{ background:var(--bg3); color:var(--t0); }}
    #theme-btn {{ padding:5px 8px; }}

    /* LAYOUT */
    #root {{ display:flex; height:calc(100vh - 48px); margin-top:48px; overflow:hidden; }}
    #map {{ flex:1 1 auto; min-width:0; position:relative; background:var(--map-bg); }}

    /* INFO PANEL */
    #info-panel {{
      position:absolute; top:14px; left:14px; z-index:1500;
      background:var(--bg1); border:1px solid var(--bd); border-radius:7px;
      padding:14px 16px; min-width:224px; max-width:268px;
      font-size:12px; box-shadow:var(--shadow); display:none;
    }}
    .ip-head {{
      font-size:13px; font-weight:600; color:var(--t0);
      margin-bottom:10px; padding-bottom:8px; border-bottom:1px solid var(--bd);
      display:flex; justify-content:space-between; align-items:center;
    }}
    .ip-x {{ background:none; border:none; color:var(--t2); cursor:pointer; font-size:14px; line-height:1; padding:0; }}
    .ip-x:hover {{ color:var(--t1); }}
    .ip-row {{ display:flex; justify-content:space-between; align-items:baseline; gap:8px; margin:5px 0; }}
    .ip-lbl {{ color:var(--t2); font-size:11px; }}
    .ip-val {{ color:var(--t0); font-weight:500; font-size:12px; text-align:right; }}
    .cls-deciduous {{ color:var(--amber) !important; }}
    .cls-evergreen {{ color:var(--sage)  !important; }}
    .cls-uncertain {{ color:var(--t1)   !important; }}
    .ip-hr {{ border:none; border-top:1px solid var(--bd); margin:10px 0; }}
    .ip-slbl {{ font-size:10px; color:var(--t2); text-transform:uppercase; letter-spacing:.7px; margin-bottom:5px; }}
    #sp-input {{
      width:100%; padding:5px 8px; border-radius:4px;
      background:var(--bg2); border:1px solid var(--bd);
      color:var(--t0); font-size:12px; outline:none;
    }}
    #sp-input:focus {{ border-color:var(--green); }}
    #sp-save {{
      width:100%; margin-top:6px; padding:5px 0; border-radius:4px;
      border:1px solid var(--bd); background:var(--bg2); color:var(--green);
      cursor:pointer; font-size:11px; font-weight:600;
    }}
    #sp-save:hover {{ background:var(--bg3); border-color:var(--green); }}

    /* RIGHT PANEL */
    #panel {{
      width:390px; min-width:290px; border-left:1px solid var(--bd);
      background:var(--bg1); display:flex; flex-direction:column; overflow:hidden;
    }}
    #panel-header {{ padding:12px 14px; border-bottom:1px solid var(--bd); flex-shrink:0; }}
    #panel-header h2 {{ font-size:13px; color:var(--t0); font-weight:600; margin-bottom:2px; }}
    #panel-hint {{ font-size:11px; color:var(--t2); }}

    /* OM selector bar */
    #om-select-section {{
      padding:10px 14px; border-bottom:1px solid var(--bd); flex-shrink:0; display:none;
    }}
    #om-select-label {{ font-size:11px; color:var(--t2); margin-bottom:6px;
      display:flex; justify-content:space-between; }}
    #om-select-label span {{ color:var(--green); font-weight:600; }}
    #om-slider {{ width:100%; accent-color:var(--green); cursor:pointer; }}
    #om-controls {{ display:flex; gap:6px; margin-top:7px; }}
    #play-btn, #reset-btn {{
      flex:1; padding:5px 0; border:1px solid var(--bd); border-radius:4px;
      background:var(--bg2); color:var(--t1); cursor:pointer; font-size:11px;
    }}
    #play-btn:hover, #reset-btn:hover {{ background:var(--bg3); color:var(--t0); }}

    /* COG tile viewer embed inside right panel */
    #tile-view-section {{
      flex:0 0 auto; display:none; position:relative; background:var(--bg2);
      border-bottom:1px solid var(--bd); width:100%; aspect-ratio:1/1;
    }}
    #tile-map {{ width:100%; height:100%; }}
    .tile-om-lbl {{
      position:absolute; top:6px; left:6px; z-index:500;
      background:rgba(0,0,0,.6); color:#eee; font-size:10px;
      padding:2px 7px; border-radius:3px; pointer-events:none;
    }}

    #content {{ flex:1 1 auto; overflow-y:auto; padding:10px 12px; }}
    .om-card {{
      border:1px solid var(--bd); border-radius:5px;
      margin-bottom:8px; background:var(--bg2); overflow:hidden;
      cursor:pointer; transition:border-color .15s;
    }}
    .om-card:hover {{ border-color:var(--t1); }}
    .om-card.active {{ border-color:var(--green); }}
    .om-meta {{ font-size:11px; color:var(--t2); padding:6px 10px;
      display:flex; justify-content:space-between; align-items:center; }}
    .om-lbl {{ color:var(--t1); font-weight:600; }}
    .om-badge {{
      font-size:10px; padding:1px 6px; border-radius:3px;
      background:var(--bg3); color:var(--t2);
    }}
    .no-data {{ color:var(--t2); font-size:12px; padding:6px 10px; font-style:italic; }}

    /* Time-series single column: crown tile maps, one per OM */
    .ts-grid {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      padding: 8px 10px;
    }}
    .ts-card {{
      border: 1px solid var(--bd);
      border-radius: 5px;
      overflow: hidden;
      background: var(--bg2);
      transition: border-color .15s;
    }}
    .ts-card.leaf-on  {{ border-color: #4caf50; }}
    .ts-card.leaf-off {{ border-color: #f44336; }}
    .ts-lbl {{
      font-size: 10px; color: var(--t2); padding: 4px 8px;
      border-bottom: 1px solid var(--bd); background: var(--bg3);
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .ts-lbl em {{ color: var(--green); font-style: normal; font-weight: 600; }}
    .ts-lbl.leaf-on  em {{ color: #4caf50; }}
    .ts-lbl.leaf-off em {{ color: #f44336; }}
    .ts-map {{
      width: 100%;
      aspect-ratio: 1 / 1;
      display: block;
      background: var(--bg0);
    }}
    .sp-row {{ padding:0 10px 8px; display:flex; gap:5px; }}
    .sp-inp {{
      flex:1; padding:4px 8px; border-radius:4px;
      background:var(--bg2); border:1px solid var(--bd); color:var(--t0); font-size:11px; outline:none;
    }}
    .sp-inp:focus {{ border-color:var(--green); }}
    .sp-btn {{
      padding:4px 10px; border-radius:4px; border:1px solid var(--bd);
      background:var(--bg2); color:var(--green); cursor:pointer; font-size:11px;
    }}

    /* COMPARISON PANEL */
    #cmp-panel {{
      width:0; overflow:hidden; transition:width .22s ease;
      border-left:1px solid var(--bd); background:var(--bg1); flex-shrink:0;
    }}
    #cmp-panel.open {{ width:580px; min-width:580px; overflow-y:auto; }}
    #cmp-inner {{ width:580px; padding:14px; }}
    #cmp-header {{
      display:flex; justify-content:space-between; align-items:center;
      margin-bottom:12px; padding-bottom:10px; border-bottom:1px solid var(--bd);
    }}
    #cmp-header h3 {{ font-size:13px; color:var(--t0); font-weight:600; }}
    #cmp-close {{ border:none; background:none; color:var(--t2); cursor:pointer; font-size:15px; line-height:1; }}
    #cmp-slots {{ display:flex; gap:10px; }}
    .cmp-slot {{ flex:1; border:1px solid var(--bd); border-radius:6px; background:var(--bg2); overflow:hidden; }}
    .cmp-slot-hd {{ padding:8px 10px; border-bottom:1px solid var(--bd); background:var(--bg1); }}
    .cmp-slot-lbl {{ font-size:10px; color:var(--t2); text-transform:uppercase; letter-spacing:.7px; margin-bottom:3px; }}
    .cmp-slot-id {{ font-size:12px; color:var(--green); font-weight:600; }}
    .cmp-info {{ padding:8px 10px; border-bottom:1px solid var(--bd); font-size:11px; }}
    .cmp-row {{ display:flex; justify-content:space-between; gap:6px; padding:2px 0; }}
    .cmp-key {{ color:var(--t2); }}
    .cmp-val {{ color:var(--t0); font-weight:500; text-align:right; }}

    /* Per-slot embedded tile map */
    .cmp-tile-map-wrap {{
      height:200px; position:relative;
      border-bottom:1px solid var(--bd);
    }}
    .cmp-tile-map {{ width:100%; height:100%; }}
    .cmp-tl {{ padding:0 10px 10px; }}
    .cmp-tl-top {{ font-size:10px; color:var(--t2); display:flex; justify-content:space-between; margin-bottom:3px; }}
    .cmp-tl-date {{ color:var(--green); font-weight:600; }}
    .cmp-slider {{ width:100%; accent-color:var(--green); cursor:pointer; margin-bottom:5px; }}
    .cmp-controls {{ display:flex; gap:5px; }}
    .cmp-play-btn, .cmp-reset-btn {{
      flex:1; padding:4px 0; border:1px solid var(--bd); border-radius:4px;
      background:var(--bg1); color:var(--t1); cursor:pointer; font-size:10px;
    }}
    .cmp-play-btn:hover, .cmp-reset-btn:hover {{ background:var(--bg3); color:var(--t0); }}
    .btn-set-slot {{
      width:calc(100% - 20px); margin:0 10px 10px;
      padding:5px 0; border:1px solid var(--bd); border-radius:4px;
      background:var(--bg1); color:var(--t1); cursor:pointer; font-size:11px;
    }}
    .btn-set-slot:hover {{ background:var(--bg3); color:var(--t0); border-color:var(--green); }}

    ::-webkit-scrollbar {{ width:4px; }}
    ::-webkit-scrollbar-track {{ background:transparent; }}
    ::-webkit-scrollbar-thumb {{ background:var(--bd); border-radius:2px; }}
  </style>
</head>
<body>

<div id="topbar">
  <span class="logo"><em>⬡</em> Interactive Visualizer</span>
  <div class="divider"></div>
  <div id="search-wrap">
    <input id="search-input" type="text" placeholder="Crown ID or species…"/>
    <button id="search-btn">Search</button>
  </div>
  <div class="divider"></div>
  <div id="filter-wrap">
    <span class="flabel">Show:</span>
    <span class="pill active" data-filter="all">All</span>
    <span class="pill" data-filter="deciduous">Deciduous</span>
    <span class="pill" data-filter="evergreen">Evergreen</span>
    <span class="pill" data-filter="uncertain">Uncertain</span>
  </div>
  <div id="crown-count-badge">— crowns</div>
  <div style="margin-left:auto;display:flex;gap:7px;align-items:center;">
    <button class="topbar-btn" id="theme-btn" title="Toggle theme">
      <svg id="ico-moon" width="15" height="15" viewBox="0 0 24 24"
           fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
      </svg>
      <svg id="ico-sun" width="15" height="15" viewBox="0 0 24 24" style="display:none;"
           fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="5"/>
        <line x1="12" y1="1"  x2="12" y2="3"/>   <line x1="12" y1="21" x2="12" y2="23"/>
        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/>  <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/>
        <line x1="1" y1="12" x2="3" y2="12"/>    <line x1="21" y1="12" x2="23" y2="12"/>
        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/> <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
      </svg>
    </button>
    <button class="topbar-btn" id="cmp-toggle">Compare crowns</button>
  </div>
</div>

<div id="root">

  <div id="map">
    <div id="info-panel">
      <div class="ip-head">
        <span id="ip-id">—</span>
        <button class="ip-x" id="ip-close">✕</button>
      </div>
      <div class="ip-row"><span class="ip-lbl">Phenology</span><span class="ip-val" id="ip-pheno">—</span></div>
      <div class="ip-row"><span class="ip-lbl">Deciduous score</span><span class="ip-val" id="ip-score">—</span></div>
      <div class="ip-row"><span class="ip-lbl">GCC amplitude</span><span class="ip-val" id="ip-gcc">—</span></div>
      <div class="ip-row"><span class="ip-lbl">Veg amplitude</span><span class="ip-val" id="ip-veg">—</span></div>
      <div class="ip-row"><span class="ip-lbl">Leaf-on OM</span><span class="ip-val" id="ip-leafon">—</span></div>
      <div class="ip-row"><span class="ip-lbl">Leaf-off OM</span><span class="ip-val" id="ip-leafoff">—</span></div>
      <hr class="ip-hr"/>
      <div class="ip-slbl">Species annotation</div>
      <input id="sp-input" type="text" placeholder="e.g. Acacia robusta"/>
      <button id="sp-save">Save species</button>
    </div>
  </div>

  <div id="panel">
    <div id="panel-header">
      <h2 id="panel-title">Crown Observations</h2>
      <div id="panel-hint">Click a crown on the map to explore.</div>
    </div>

    <!-- OM timeline slider -->
    <div id="om-select-section">
      <div id="om-select-label">
        <span>Observation month</span>
        <span id="om-cur-label"></span>
      </div>
      <input id="om-slider" type="range" min="0" value="0" step="1"/>
      <div id="om-controls">
        <button id="play-btn">▶  Animate</button>
        <button id="reset-btn">↺  Reset</button>
      </div>
    </div>

    <!-- Embedded tile map showing just this crown's area -->
    <div id="tile-view-section">
      <div id="tile-map"></div>
      <div class="tile-om-lbl" id="tile-om-lbl"></div>
    </div>

    <div id="content">
      <div class="no-data" style="padding:24px 14px;">
        Click any crown on the map to explore its observations.
      </div>
    </div>
  </div>

  <div id="cmp-panel">
    <div id="cmp-inner">
      <div id="cmp-header">
        <h3>Crown Comparison</h3>
        <button id="cmp-close">✕</button>
      </div>
      <div id="cmp-slots">

        <div class="cmp-slot" id="slot-a">
          <div class="cmp-slot-hd">
            <div class="cmp-slot-lbl">Crown A</div>
            <div class="cmp-slot-id" id="slot-a-id">— not set —</div>
          </div>
          <div class="cmp-info" id="slot-a-info" style="display:none;"></div>
          <div class="cmp-tile-map-wrap"><div class="cmp-tile-map" id="cmap-a"></div></div>
          <div class="cmp-tl" id="slot-a-tl" style="display:none;">
            <div class="cmp-tl-top"><span>Observation</span>
              <span class="cmp-tl-date" id="slot-a-tl-date"></span></div>
            <input class="cmp-slider" id="slider-a" type="range" min="0" value="0" step="1"/>
            <div class="cmp-controls">
              <button class="cmp-play-btn" id="play-a">▶ Play</button>
              <button class="cmp-reset-btn" id="reset-a">↺ Reset</button>
            </div>
          </div>
          <button class="btn-set-slot" onclick="assignSlot('a')">Set to selected crown</button>
        </div>

        <div class="cmp-slot" id="slot-b">
          <div class="cmp-slot-hd">
            <div class="cmp-slot-lbl">Crown B</div>
            <div class="cmp-slot-id" id="slot-b-id">— not set —</div>
          </div>
          <div class="cmp-info" id="slot-b-info" style="display:none;"></div>
          <div class="cmp-tile-map-wrap"><div class="cmp-tile-map" id="cmap-b"></div></div>
          <div class="cmp-tl" id="slot-b-tl" style="display:none;">
            <div class="cmp-tl-top"><span>Observation</span>
              <span class="cmp-tl-date" id="slot-b-tl-date"></span></div>
            <input class="cmp-slider" id="slider-b" type="range" min="0" value="0" step="1"/>
            <div class="cmp-controls">
              <button class="cmp-play-btn" id="play-b">▶ Play</button>
              <button class="cmp-reset-btn" id="reset-b">↺ Reset</button>
            </div>
          </div>
          <button class="btn-set-slot" onclick="assignSlot('b')">Set to selected crown</button>
        </div>

      </div>
      <p style="font-size:11px;color:var(--t2);margin-top:12px;line-height:1.6;">
        Select a crown, assign to A or B. Each slot has its own independent timeline.<br/>
        The embedded map loads only the tiles needed for the crown's area.
      </p>
    </div>
  </div>

</div><!-- /root -->

<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
        integrity="sha256-20nQCchB9co0qIjJZRGuk2/Z9VM+kNiyxNV1lvTlZBo=" crossorigin=""></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet.draw/1.0.4/leaflet.draw.js"></script>
<script>
// ═══════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════
const RUN_TAG      = {json.dumps(run_tag)};
const GEOJSON      = {geojson_inline};
const MANIFEST     = {manifest_inline};
const PHENOLOGY    = {pheno_inline};
const OM_TILES     = {om_tile_manifest};   // [{{om_id, stem, tile_url, bounds:{{w,s,e,n}}, min_zoom, max_zoom}}, ...]
const UNDERLAY_OM  = {underlay_om_id};

// Build stem → tile-info lookup
const OM_BY_ID  = {{}};   // om_id (int) → tile entry
const OM_STEMS  = {{}};   // om_id → stem  (for leaf-on/off display)
OM_TILES.forEach(o => {{ OM_BY_ID[o.om_id] = o; OM_STEMS[o.om_id] = o.stem; }});
const orderedOMs = OM_TILES.map(o => o.om_id); // [1,2,...,N]

// ═══════════════════════════════════════════════════════
// THEME
// ═══════════════════════════════════════════════════════
const htmlEl  = document.documentElement;
const icoMoon = document.getElementById('ico-moon');
const icoSun  = document.getElementById('ico-sun');
(function() {{
  const t = localStorage.getItem('iv-theme') || 'dark';
  if (t === 'light') {{
    htmlEl.dataset.theme = 'light';
    icoMoon.style.display = 'none';
    icoSun.style.display  = 'block';
  }}
}})();
document.getElementById('theme-btn').addEventListener('click', () => {{
  const next = htmlEl.dataset.theme === 'dark' ? 'light' : 'dark';
  htmlEl.dataset.theme = next;
  icoMoon.style.display = next === 'dark'  ? 'block' : 'none';
  icoSun.style.display  = next === 'light' ? 'block' : 'none';
  localStorage.setItem('iv-theme', next);
}});

// ═══════════════════════════════════════════════════════
// SPECIES
// ═══════════════════════════════════════════════════════
const SP_KEY = 'iv-species-' + RUN_TAG;
let speciesMap = JSON.parse(localStorage.getItem(SP_KEY) || '{{}}');
const saveSpecies = (id,v) => {{ speciesMap[id]=v; localStorage.setItem(SP_KEY,JSON.stringify(speciesMap)); }};
const getSpecies  = id => speciesMap[id] || '';

// ═══════════════════════════════════════════════════════
// MAIN MAP  (geographic CRS, WGS84 / Web Mercator tiles)
// ═══════════════════════════════════════════════════════
const underlayInfo = OM_BY_ID[UNDERLAY_OM];
const mapBounds = underlayInfo
  ? [[underlayInfo.bounds.s, underlayInfo.bounds.w],
     [underlayInfo.bounds.n, underlayInfo.bounds.e]]
  : [[-90,-180],[90,180]];

const map = L.map('map', {{
  crs: L.CRS.EPSG3857,
  zoomControl: false,
  attributionControl: false,
  minZoom: underlayInfo ? underlayInfo.min_zoom : 0,  // can't zoom out past OM extent
  maxZoom: 22,
}});
L.control.zoom({{ position: 'topright' }}).addTo(map);

// Underlay tile layer — switches when user changes OM in right panel
let currentMainTileLayer = null;

function setMainTileLayer(omId) {{
  const info = OM_BY_ID[omId];
  if (!info) return;
  if (currentMainTileLayer) map.removeLayer(currentMainTileLayer);
  currentMainTileLayer = L.tileLayer(info.tile_url, {{
    tms: false,
    minZoom: info.min_zoom,
    maxZoom: 22,           // allow over-zoom (Leaflet stretches tiles past maxNativeZoom)
    maxNativeZoom: info.max_zoom,  // highest zoom we actually have tiles for
    opacity: 1,
    attribution: '',
    errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
  }}).addTo(map);
}}

setMainTileLayer(UNDERLAY_OM);
map.fitBounds(mapBounds);

// Lock main map zoom so OM always covers ≥60% of the viewport.
// After the initial fitBounds, the OM fills ~100% of the screen (fitZoom).
// Allowing one level out (fitZoom - 1) gives ~50% coverage.
// We lock at fitZoom - 1: the raster always fills at least ~50–60%.
// zoomend fires once after fitBounds completes.
map.once('zoomend', function() {{
  const fitZoom = map.getZoom();
  map.setMinZoom(Math.max(0, fitZoom - 1));
}});

// Draw tool
const drawnItems = new L.FeatureGroup().addTo(map);
map.addControl(new L.Control.Draw({{
  position: 'topright',
  draw: {{
    polygon:   {{ shapeOptions:{{ color:'#5a8a6a', weight:2 }} }},
    rectangle: {{ shapeOptions:{{ color:'#5a8a6a', weight:2 }} }},
    polyline:false, circle:false, marker:false, circlemarker:false,
  }},
  edit: {{ featureGroup: drawnItems }},
}}));
map.on('draw:created', e => {{
  drawnItems.clearLayers(); drawnItems.addLayer(e.layer);
  const bb = e.layer.getBounds();
  const inside = [];
  geoLayer.eachLayer(lyr => {{
    if (bb.contains(lyr.getBounds().getCenter())) inside.push(lyr.feature.properties.crown_label);
  }});
  if (inside.length)
    document.getElementById('content').innerHTML =
      `<div style="padding:14px;">
        <div style="font-size:12px;color:var(--t1);font-weight:600;margin-bottom:6px;">
          ${{inside.length}} crown${{inside.length>1?'s':''}} in selection</div>
        <div style="font-size:11px;color:var(--t2);line-height:1.8;">
          ${{inside.slice(0,80).join(', ')}}${{inside.length>80?' …':''}}</div></div>`;
}});

// ═══════════════════════════════════════════════════════
// PHENOLOGY HELPERS
// ═══════════════════════════════════════════════════════
const getPhenoClass = lbl => {{
  const d=PHENOLOGY[lbl]; if(!d) return 'uncertain';
  return d.phenology_class==='unknown'?'uncertain':(d.phenology_class||'uncertain');
}};
const getPhenoColor = cls =>
  cls==='deciduous'?'#b89550':cls==='evergreen'?'#5f9e6a':'#525a6e';

// ═══════════════════════════════════════════════════════
// CROWN LAYER (GeoJSON — lat/lon coordinates)
// ═══════════════════════════════════════════════════════
let selLayer=null, curCrownId=null, curFeature=null, activeFilter='all';

const styleDef = f => {{
  return {{color:'white',weight:3,fillOpacity:0,fillColor:'transparent'}}; }};
const styleHov = f => {{
  return {{color:'white',weight:4,fillOpacity:0,fillColor:'transparent'}}; }};
const styleSel = f => {{
  return {{color:'white',weight:5,fillOpacity:0,fillColor:'transparent'}}; }};

const geoLayer = L.geoJSON(GEOJSON, {{
  style: styleDef,
  onEachFeature(feature, layer) {{
    layer.on({{
      mouseover() {{ if(layer!==selLayer){{layer.setStyle(styleHov(feature));layer.bringToFront();}} }},
      mouseout()  {{ if(layer!==selLayer) layer.setStyle(styleDef(feature)); }},
      click(e) {{
        L.DomEvent.stopPropagation(e);
        if(selLayer&&selLayer!==layer) selLayer.setStyle(styleDef(selLayer.feature));
        if(selLayer===layer) {{
          selLayer.setStyle(styleDef(feature)); selLayer=null; curCrownId=null; curFeature=null;
          document.getElementById('info-panel').style.display='none'; return;
        }}
        selLayer=layer; layer.setStyle(styleSel(feature)); layer.bringToFront();
        curCrownId=feature.properties.crown_label; curFeature=feature;
        showInfoPanel(feature); loadPanel(feature);
      }}
    }});
  }}
}}).addTo(map);

const numCrowns = MANIFEST.num_crowns || GEOJSON.features.length;
document.getElementById('crown-count-badge').textContent = numCrowns + ' crowns';

// ═══════════════════════════════════════════════════════
// INFO PANEL
// ═══════════════════════════════════════════════════════
function showInfoPanel(feature) {{
  const lbl=feature.properties.crown_label, ph=PHENOLOGY[lbl]||{{}}, cls=getPhenoClass(lbl);
  document.getElementById('ip-id').textContent=lbl;
  const el=document.getElementById('ip-pheno');
  el.textContent=cls[0].toUpperCase()+cls.slice(1); el.className='ip-val cls-'+cls;
  document.getElementById('ip-score').textContent=ph.deciduous_score!=null?ph.deciduous_score+'%':'—';
  document.getElementById('ip-gcc').textContent=ph.gcc_amplitude!=null?ph.gcc_amplitude.toFixed(3):'—';
  document.getElementById('ip-veg').textContent=ph.veg_amplitude!=null?ph.veg_amplitude.toFixed(3):'—';
  // Leaf-on with stem name
  if(ph.leaf_on_om) {{
    const st=OM_STEMS[ph.leaf_on_om]||'';
    document.getElementById('ip-leafon').textContent='OM'+String(ph.leaf_on_om).padStart(2,'0')+(st?' — '+st:'');
  }} else document.getElementById('ip-leafon').textContent='—';
  if(ph.leaf_off_om) {{
    const st=OM_STEMS[ph.leaf_off_om]||'';
    document.getElementById('ip-leafoff').textContent='OM'+String(ph.leaf_off_om).padStart(2,'0')+(st?' — '+st:'');
  }} else document.getElementById('ip-leafoff').textContent='—';
  document.getElementById('sp-input').value=getSpecies(lbl);
  document.getElementById('info-panel').style.display='block';
}}

document.getElementById('ip-close').addEventListener('click', () => {{
  document.getElementById('info-panel').style.display='none';
  if(selLayer){{selLayer.setStyle(styleDef(selLayer.feature));selLayer=null;}}
  curCrownId=null; curFeature=null;
}});
document.getElementById('sp-save').addEventListener('click', function() {{
  if(!curCrownId) return;
  saveSpecies(curCrownId, document.getElementById('sp-input').value.trim());
  this.textContent='✓ Saved'; setTimeout(()=>{{this.textContent='Save species';}},1500);
  applyFilter();
}});

// ═══════════════════════════════════════════════════════
// RIGHT PANEL — COG tile viewer per crown
// ═══════════════════════════════════════════════════════
const omSlider = document.getElementById('om-slider');
const playBtn  = document.getElementById('play-btn');
let playTimer  = null;
let tileMap    = null;   // Leaflet map inside #tile-map
let tileLayer  = null;   // current tile layer on tileMap
let curCrownBounds = null;

function initTileMap() {{
  if (tileMap) return;
  tileMap = L.map('tile-map', {{
    crs: L.CRS.EPSG3857,
    zoomControl:      false,
    attributionControl: false,
    dragging:         false,
    scrollWheelZoom:  false,
    doubleClickZoom:  false,
    touchZoom:        false,
    keyboard:         false,
    boxZoom:          false,
  }});
}}

function loadPanel(feature) {{
  const lbl=feature.properties.crown_label;
  const ph=PHENOLOGY[lbl]||{{}}, cls=getPhenoClass(lbl);
  document.getElementById('panel-title').innerHTML=
    `<span style="color:${{getPhenoColor(cls)}};margin-right:5px;">◆</span>${{lbl}}`;
  document.getElementById('panel-hint').textContent=
    cls[0].toUpperCase()+cls.slice(1)+
    (ph.deciduous_score!=null?'  ·  score '+ph.deciduous_score+'%':'')+
    (getSpecies(lbl)?'  ·  '+getSpecies(lbl):'');

  // Crown bounds from GeoJSON (lat/lon)
  const lyrBounds = selLayer ? selLayer.getBounds() : null;
  curCrownBounds  = lyrBounds;

  // Show OM slider
  omSlider.max = orderedOMs.length - 1; omSlider.value = UNDERLAY_OM - 1;
  document.getElementById('om-select-section').style.display = 'block';
  document.getElementById('tile-view-section').style.display = 'block';

  updateOmLabel(UNDERLAY_OM - 1);

  // Init or refresh the embedded tile map
  initTileMap();
  showCrownOnTileMap(UNDERLAY_OM, lyrBounds);

  // OM card list in content
  renderOMCards(lbl);
}}

// Registry of per-OM mini Leaflet maps inside the time-series grid.
// Keyed by om_id. Destroyed and rebuilt each time a new crown is selected.
let tsMaps = {{}};

function destroyTsMaps() {{
  Object.values(tsMaps).forEach(m => {{ try {{ m.remove(); }} catch(e) {{}} }});
  tsMaps = {{}};
}}

function renderOMCards(lbl) {{
  const sp = getSpecies(lbl);
  const ph = PHENOLOGY[lbl] || {{}};

  // Species row + single-column time-series grid (non-interactive)
  let html = `
    <div class="sp-row">
      <input class="sp-inp" id="panel-sp" type="text"
             placeholder="Species…" value="${{sp}}"/>
      <button class="sp-btn" onclick="savePanelSp('${{lbl}}')">Save</button>
    </div>
    <div class="ts-grid" id="ts-grid">`;

  OM_TILES.forEach(om => {{
    const omLabel = 'OM' + String(om.om_id).padStart(2, '0');
    const isOn    = ph.leaf_on_om  === om.om_id;
    const isOff   = ph.leaf_off_om === om.om_id;
    const cardCls = ['ts-card',
                     isOn  ? 'leaf-on'  : '',
                     isOff ? 'leaf-off' : ''].filter(Boolean).join(' ');
    const lblCls  = ['ts-lbl',
                     isOn  ? 'leaf-on'  : '',
                     isOff ? 'leaf-off' : ''].filter(Boolean).join(' ');
    html += `
      <div class="${{cardCls}}" id="ts-card-${{om.om_id}}">
        <div class="${{lblCls}}">
          <em>${{omLabel}}</em> ${{om.stem}}
        </div>
        <div class="ts-map" id="ts-map-${{om.om_id}}"></div>
      </div>`;
  }});

  html += `</div>`;
  document.getElementById('content').innerHTML = html;

  // Initialise a Leaflet map inside each ts-map div.
  // Must run after the HTML is in the DOM — use setTimeout for paint tick.
  if (!curCrownBounds) return;
  const sw  = curCrownBounds.getSouthWest();
  const ne  = curCrownBounds.getNorthEast();
  const buf = 0.00005;
  const tight = L.latLngBounds(
    [sw.lat - buf, sw.lng - buf],
    [ne.lat + buf, ne.lng + buf]
  );

  // Destroy old mini-maps from previous crown before building new ones
  destroyTsMaps();

  setTimeout(() => {{
    OM_TILES.forEach(om => {{
      const el = document.getElementById('ts-map-' + om.om_id);
      if (!el) return;

      const m = L.map(el, {{
        crs:              L.CRS.EPSG3857,
        zoomControl:      false,
        attributionControl: false,
        dragging:         false,    // thumbnail — panning disabled
        scrollWheelZoom:  false,
        doubleClickZoom:  false,
        touchZoom:        false,
        keyboard:         false,
      }});

      L.tileLayer(om.tile_url, {{
        tms:           false,
        minZoom:       om.min_zoom,
        maxZoom:       22,
        maxNativeZoom: om.max_zoom,
        opacity:       1,
        attribution:   '',
        errorTileUrl:  'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
      }}).addTo(m);

      m.fitBounds(tight, {{ animate: false }});

      // Lock: cannot zoom out past the initial fitted view
      m.once('zoomend', function() {{
        m.setMinZoom(m.getZoom());
        m.setMaxZoom(22);
      }});

      tsMaps[om.om_id] = m;
    }});
  }}, 30);
}}

function selectOM(idx) {{
  // This function is no longer called since cards are not clickable.
  // The slider still controls the large tile map via omSlider.addEventListener.
  omSlider.value = idx;
  updateOmLabel(idx);
  const omId = orderedOMs[idx];
  // The slider listener in original code handles the tile map update.
}}

function updateOmLabel(idx) {{
  const om=OM_TILES[idx];
  document.getElementById('om-cur-label').textContent=
    om?'OM'+String(om.om_id).padStart(2,'0')+' — '+om.stem:'';
  document.getElementById('tile-om-lbl').textContent=
    om?'OM'+String(om.om_id).padStart(2,'0')+' — '+om.stem:'';
}}

function showCrownOnTileMap(omId, crownBounds) {{
  const info = OM_BY_ID[omId]; if(!info) return;
  if(tileLayer) tileMap.removeLayer(tileLayer);
  tileLayer = L.tileLayer(info.tile_url, {{
    tms: false,
    minZoom: info.min_zoom,
    maxZoom: 22,                          // allow over-zoom past native tiles
    maxNativeZoom: info.max_zoom,         // highest zoom with actual tiles
    opacity: 1, attribution: '',
    errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
  }}).addTo(tileMap);
  if(crownBounds) {{
    // Fixed geographic buffer: ~0.00015° ≈ 15m — just enough context, not a huge area
    const sw = crownBounds.getSouthWest();
    const ne = crownBounds.getNorthEast();
    const buf = 0.00015;  // degrees — ~15 metres at this latitude
    const tight = L.latLngBounds(
      [sw.lat - buf, sw.lng - buf],
      [ne.lat + buf, ne.lng + buf]
    );
    tileMap.fitBounds(tight, {{ animate: false }});

    // Lock: cannot zoom out past the initial crown view.
    // zoomend fires once after fitBounds resolves the zoom level.
    tileMap.once('zoomend', function() {{
      tileMap.setMinZoom(tileMap.getZoom());
    }});
  }}
}}

omSlider.addEventListener('input', function() {{
  const idx=parseInt(this.value);
  updateOmLabel(idx);
  const omId=orderedOMs[idx];
  // Update crown tile view only — DO NOT touch main map
  if(curCrownBounds) showCrownOnTileMap(omId, curCrownBounds);
  if(curCrownId) renderOMCards(curCrownId);
}});

playBtn.addEventListener('click', function() {{
  const tsGrid = document.getElementById('ts-grid');
  if(playTimer){{
    clearInterval(playTimer); playTimer=null;
    this.textContent='▶  Animate';
    if(tsGrid) tsGrid.style.display='';   // restore time-series
    return;
  }}
  this.textContent='⏸  Pause';
  if(tsGrid) tsGrid.style.display='none'; // hide time-series during playback
  let i=parseInt(omSlider.value);
  playTimer=setInterval(()=>{{
    i=(i+1)%orderedOMs.length; omSlider.value=i; omSlider.dispatchEvent(new Event('input'));
    if(i===orderedOMs.length-1){{
      clearInterval(playTimer); playTimer=null;
      playBtn.textContent='▶  Animate';
      if(tsGrid) tsGrid.style.display='';  // restore when animation ends
    }}
  }},800);
}});
document.getElementById('reset-btn').addEventListener('click',()=>{{
  if(playTimer){{clearInterval(playTimer);playTimer=null;playBtn.textContent='▶  Animate';}}
  omSlider.value=UNDERLAY_OM-1; omSlider.dispatchEvent(new Event('input'));
}});

function savePanelSp(lbl) {{
  saveSpecies(lbl,document.getElementById('panel-sp').value.trim());
  if(curCrownId===lbl) document.getElementById('sp-input').value=getSpecies(lbl);
  applyFilter();
}}

// ═══════════════════════════════════════════════════════
// COMPARISON PANEL — independent tile maps per slot
// ═══════════════════════════════════════════════════════
document.getElementById('cmp-toggle').addEventListener('click',()=>
  document.getElementById('cmp-panel').classList.toggle('open'));
document.getElementById('cmp-close').addEventListener('click',()=>
  document.getElementById('cmp-panel').classList.remove('open'));

let slotData={{a:null,b:null}};
let slotMaps={{a:null,b:null}};
let slotLayers={{a:null,b:null}};
let slotTimers={{a:null,b:null}};

function initSlotMap(s) {{
  if(slotMaps[s]) return;
  slotMaps[s] = L.map('cmap-'+s, {{
    crs: L.CRS.EPSG3857, zoomControl:false, attributionControl:false,
    dragging:true, scrollWheelZoom:true,
  }});
}}

function assignSlot(s) {{
  if(!curCrownId||!curFeature){{alert('Select a crown first.');return;}}
  initSlotMap(s);
  slotData[s]={{label:curCrownId, feature:curFeature, bounds:curCrownBounds, omIdx:parseInt(omSlider.value)}};
  renderSlot(s);
  document.getElementById('cmp-panel').classList.add('open');
  // Invalidate map size after panel opens
  setTimeout(()=>{{if(slotMaps[s]) slotMaps[s].invalidateSize();}}, 250);
}}

function renderSlot(s) {{
  const d=slotData[s]; if(!d) return;
  const ph=PHENOLOGY[d.label]||{{}}, cls=getPhenoClass(d.label);
  document.getElementById('slot-'+s+'-id').textContent=d.label;

  // Info table
  let info='';
  info+=`<div class="cmp-row"><span class="cmp-key">Phenology</span>
    <span class="cmp-val cls-${{cls}}">${{cls[0].toUpperCase()+cls.slice(1)}}</span></div>`;
  if(ph.deciduous_score!=null)
    info+=`<div class="cmp-row"><span class="cmp-key">Score</span><span class="cmp-val">${{ph.deciduous_score}}%</span></div>`;
  if(ph.gcc_amplitude!=null)
    info+=`<div class="cmp-row"><span class="cmp-key">GCC amplitude</span><span class="cmp-val">${{ph.gcc_amplitude.toFixed(3)}}</span></div>`;
  if(ph.veg_amplitude!=null)
    info+=`<div class="cmp-row"><span class="cmp-key">Veg amplitude</span><span class="cmp-val">${{ph.veg_amplitude.toFixed(3)}}</span></div>`;
  if(ph.leaf_on_om) {{
    const st=OM_STEMS[ph.leaf_on_om]||'';
    const str='OM'+String(ph.leaf_on_om).padStart(2,'0')+(st?' — '+st:'');
    info+=`<div class="cmp-row"><span class="cmp-key">Leaf-on</span><span class="cmp-val">${{str}}</span></div>`;
  }}
  if(ph.leaf_off_om) {{
    const st=OM_STEMS[ph.leaf_off_om]||'';
    const str='OM'+String(ph.leaf_off_om).padStart(2,'0')+(st?' — '+st:'');
    info+=`<div class="cmp-row"><span class="cmp-key">Leaf-off</span><span class="cmp-val">${{str}}</span></div>`;
  }}
  const sp=getSpecies(d.label);
  if(sp) info+=`<div class="cmp-row"><span class="cmp-key">Species</span><span class="cmp-val">${{sp}}</span></div>`;
  const infoEl=document.getElementById('slot-'+s+'-info');
  infoEl.innerHTML=info; infoEl.style.display='block';

  // Setup slider
  const slider=document.getElementById('slider-'+s);
  slider.max=orderedOMs.length-1; slider.value=d.omIdx;
  document.getElementById('slot-'+s+'-tl').style.display='block';
  showSlotTile(s, d.omIdx);
  updateSlotDate(s, d.omIdx);
}}

function showSlotTile(s, idx) {{
  const d=slotData[s]; if(!d||!slotMaps[s]) return;
  const omId=orderedOMs[idx], info=OM_BY_ID[omId]; if(!info) return;
  if(slotLayers[s]) slotMaps[s].removeLayer(slotLayers[s]);
  slotLayers[s]=L.tileLayer(info.tile_url,{{
    tms:false, minZoom:info.min_zoom, maxZoom:22, maxNativeZoom:info.max_zoom,
    opacity:1, attribution:'',
    errorTileUrl:'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
  }}).addTo(slotMaps[s]);
  if(d.bounds) {{
    const sb=d.bounds, buf=0.00015;
    slotMaps[s].fitBounds(
      L.latLngBounds([sb.getSouthWest().lat-buf,sb.getSouthWest().lng-buf],
                     [sb.getNorthEast().lat+buf,sb.getNorthEast().lng+buf]),
      {{animate:false}}
    );
  }}
}}

function updateSlotDate(s,idx) {{
  const om=OM_TILES[idx];
  document.getElementById('slot-'+s+'-tl-date').textContent=
    om?'OM'+String(om.om_id).padStart(2,'0')+' — '+om.stem:'';
}}

['a','b'].forEach(s=>{{
  document.getElementById('slider-'+s).addEventListener('input',function(){{
    const idx=parseInt(this.value); showSlotTile(s,idx); updateSlotDate(s,idx);
  }});
  document.getElementById('play-'+s).addEventListener('click',function(){{
    if(slotTimers[s]){{clearInterval(slotTimers[s]);slotTimers[s]=null;this.textContent='▶ Play';return;}}
    this.textContent='⏸ Pause';
    const slider=document.getElementById('slider-'+s); let i=parseInt(slider.value);
    slotTimers[s]=setInterval(()=>{{
      i=(i+1)%orderedOMs.length; slider.value=i; showSlotTile(s,i); updateSlotDate(s,i);
      if(i===orderedOMs.length-1){{clearInterval(slotTimers[s]);slotTimers[s]=null;
        document.getElementById('play-'+s).textContent='▶ Play';}}
    }},800);
  }});
  document.getElementById('reset-'+s).addEventListener('click',()=>{{
    if(slotTimers[s]){{clearInterval(slotTimers[s]);slotTimers[s]=null;
      document.getElementById('play-'+s).textContent='▶ Play';}}
    const slider=document.getElementById('slider-'+s);
    slider.value=0; showSlotTile(s,0); updateSlotDate(s,0);
  }});
}});

// ═══════════════════════════════════════════════════════
// SEARCH
// ═══════════════════════════════════════════════════════
document.getElementById('search-btn').addEventListener('click',doSearch);
document.getElementById('search-input').addEventListener('keydown',e=>{{if(e.key==='Enter')doSearch();}});
function doSearch(){{
  const q=document.getElementById('search-input').value.trim().toLowerCase(); if(!q) return;
  let found=null;
  geoLayer.eachLayer(lyr=>{{
    if(found) return;
    const lbl=(lyr.feature.properties.crown_label||'').toLowerCase();
    if(lbl.includes(q)||String(lyr.feature.properties.crown_index)===q) found=lyr;
  }});
  if(!found){{
    const matches=[];
    geoLayer.eachLayer(lyr=>{{
      if(getSpecies(lyr.feature.properties.crown_label).toLowerCase().includes(q)) matches.push(lyr);
    }});
    if(matches.length){{
      map.fitBounds(L.featureGroup(matches).getBounds().pad(.3));
      matches.forEach(lyr=>{{lyr.setStyle(styleHov(lyr.feature));lyr.bringToFront();}});
      document.getElementById('content').innerHTML=
        `<div style="padding:14px;">
          <div style="font-size:12px;color:var(--t1);font-weight:600;margin-bottom:6px;">
            ${{matches.length}} crown${{matches.length>1?'s':''}} tagged as "${{q}}"</div>
          <div style="font-size:11px;color:var(--t2);line-height:1.8;">
            ${{matches.map(l=>l.feature.properties.crown_label).join(', ')}}</div>
        </div>`;
      return;
    }}
    document.getElementById('content').innerHTML=
      `<div class="no-data" style="padding:16px;">No match for "${{q}}".</div>`;
    return;
  }}
  map.fitBounds(found.getBounds().pad(1.5));
  if(selLayer&&selLayer!==found) selLayer.setStyle(styleDef(selLayer.feature));
  selLayer=found; found.setStyle(styleSel(found.feature)); found.bringToFront();
  curCrownId=found.feature.properties.crown_label; curFeature=found.feature;
  showInfoPanel(found.feature); loadPanel(found.feature);
}}

// ═══════════════════════════════════════════════════════
// FILTER
// ═══════════════════════════════════════════════════════
document.querySelectorAll('.pill').forEach(p=>{{
  p.addEventListener('click',function(){{
    document.querySelectorAll('.pill').forEach(x=>x.classList.remove('active'));
    this.classList.add('active'); activeFilter=this.dataset.filter; applyFilter();
  }});
}});
function applyFilter(){{
  let vis=0;
  geoLayer.eachLayer(lyr=>{{
    const lbl=lyr.feature.properties.crown_label, cls=getPhenoClass(lbl);
    const show=activeFilter==='all'||
      (activeFilter==='uncertain'&&(cls==='uncertain'||cls==='unknown'))||
      activeFilter===cls;
    if(show){{lyr.setStyle(styleDef(lyr.feature));lyr.bringToFront();vis++;}}
    else lyr.setStyle({{color:'transparent',fillColor:'transparent',weight:0}});
  }});
  document.getElementById('crown-count-badge').textContent=
    vis+(vis<numCrowns?' / '+numCrowns:'')+' crowns';
}}

// ═══════════════════════════════════════════════════════
// KEYBOARD
// ═══════════════════════════════════════════════════════
document.addEventListener('keydown',e=>{{
  if(e.target.tagName==='INPUT') return;
  if(e.key==='Escape') document.getElementById('ip-close').click();
  if(e.key==='c') document.getElementById('cmp-toggle').click();
  if(e.key===' '){{e.preventDefault();playBtn.click();}}
  if(e.key==='ArrowRight'){{omSlider.value=Math.min(+omSlider.value+1,+omSlider.max);omSlider.dispatchEvent(new Event('input'));}}
  if(e.key==='ArrowLeft') {{omSlider.value=Math.max(+omSlider.value-1,0);           omSlider.dispatchEvent(new Event('input'));}}
}});
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline Step 4b: Interactive Crown Phenology Viewer")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-threshold-tag",  default="conf_0p45")
    parser.add_argument("--align-threshold-tag", default="conf_0p65")
    parser.add_argument("--align-method",        default="pcc_tiled")
    parser.add_argument("--skip-if-done", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}", file=sys.stderr)
        return 1

    config        = load_config(config_path)
    project_root  = Path(config["project_root"])
    viewer_dir    = Path(config["viewer_dir"])
    tracking_dir  = Path(config["tracking_dir"])
    phenology_dir = Path(config["phenology_dir"])
    run_name      = config.get("run_name", "pipeline")
    dataset_name  = config.get("dataset_id", run_name)

    html_path = viewer_dir / "index.html"
    if args.skip_if_done and html_path.exists():
        print(f"[SKIP] Viewer already exists: {html_path}")
        return 0

    # ── Load tile manifest from Step 4a ──────────────────────────────────────
    manifest_path = Path(config.get("tile_manifest",
                                     viewer_dir / "tile_manifest.json"))
    if not manifest_path.exists():
        print(f"ERROR: tile_manifest.json not found at {manifest_path}", file=sys.stderr)
        print(f"       Run Step 4a first: python 04a_cog_tiling.py --config {config_path}",
              file=sys.stderr)
        return 1

    tile_manifest  = json.loads(manifest_path.read_text())
    om_tile_entries = tile_manifest["oms"]
    underlay_om_id  = tile_manifest["underlay_om_id"]
    num_oms         = tile_manifest["num_oms"]
    print(f"Loaded tile manifest: {num_oms} OMs, underlay = OM{underlay_om_id:02d}")

    import geopandas as gpd
    import rasterio
    from rasterio.warp import transform_bounds

    setup_app_dir(project_root)
    from tree_tracking import TreeTrackingGraph

    pairs, om_stems = build_pairs_and_om_stems(config)
    crowns_dir_path = Path(config["crowns_dir"])
    om_dir_path     = Path(config["om_dir"])

    # ── Load consensus crowns ────────────────────────────────────────────────
    consensus_gpkg = Path(config.get("consensus_gpkg",
                                     tracking_dir / "consensus_crowns_complete_all.gpkg"))
    if not consensus_gpkg.exists():
        print(f"ERROR: consensus crowns not found: {consensus_gpkg}", file=sys.stderr)
        return 1
    crowns = gpd.read_file(str(consensus_gpkg))
    crowns = crowns[crowns.geometry.notnull() & ~crowns.geometry.is_empty].reset_index(drop=True)
    print(f"Loaded {len(crowns)} consensus crowns")

    # ── Alignment ─────────────────────────────────────────────────────────────
    print("\nInitializing tracker …")
    tracker = TreeTrackingGraph(
        auto_discover=False,
        multithresh_dir=str(crowns_dir_path),
        ortho_dir=str(om_dir_path),
        output_dir=str(viewer_dir),
        simplify_tol=1.0, resize_factor=0.1, max_crowns_preview=200,
    )
    tracker.file_pairs = [(gpkg, tif) for gpkg, tif, _ in pairs]
    tracker.om_ids     = list(range(1, num_oms + 1))
    tracker.base_threshold_tag = None

    saved_shifts_raw = config.get("alignment_shifts", {})
    saved_shifts = {int(k): (float(v[0]), float(v[1]))
                    for k, v in saved_shifts_raw.items()} if saved_shifts_raw else {}

    if saved_shifts:
        print(f"  Using saved alignment shifts ({len(saved_shifts)} OMs)")
        tracker.load_multithreshold_data(base_threshold_tag=args.base_threshold_tag,
                                         load_images=False, align=False)
        tracker.alignment_shifts = saved_shifts
        from shapely.affinity import affine_transform as shapely_affine
        for om_id in tracker.om_ids:
            dx, dy = saved_shifts.get(om_id, (0.0, 0.0))
            if om_id == tracker.om_ids[0] or (dx == 0.0 and dy == 0.0):
                continue
            gdf = tracker.crowns_gdfs.get(om_id)
            if gdf is None or gdf.empty:
                continue
            params = (1.0, 0.0, 0.0, 1.0, dx, dy)
            gdf = gdf.copy()
            gdf["geometry"] = gdf["geometry"].apply(
                lambda g: shapely_affine(g, params) if g is not None else g)
            tracker.crowns_gdfs[om_id] = gdf
            tracker.crown_attrs[om_id] = [
                tracker._compute_crown_attributes(row.geometry)
                for _, row in gdf.iterrows()]
    else:
        tracker.load_multithreshold_data(base_threshold_tag=args.base_threshold_tag,
                                         load_images=False, align=True,
                                         align_method=args.align_method,
                                         align_threshold_tag=args.align_threshold_tag)

    # ── Crown GeoJSON in WGS84 ────────────────────────────────────────────────
    cogs_dir = viewer_dir / "cogs"
    ref_cog  = cogs_dir / f"OM{underlay_om_id:02d}_{pairs[underlay_om_id-1][2]}.tif"
    ref_tif  = str(ref_cog) if ref_cog.exists() else pairs[underlay_om_id - 1][1]

    with rasterio.open(ref_tif) as src:
        raster_crs = src.crs

    crowns_display = crowns.copy()
    if crowns_display.crs is None and raster_crs:
        crowns_display = crowns_display.set_crs(raster_crs, allow_override=True)
    elif raster_crs and crowns_display.crs != raster_crs:
        crowns_display = crowns_display.to_crs(raster_crs)
    crowns_wgs84 = crowns_display.to_crs("EPSG:4326")

    features = []
    for i, row in crowns_wgs84.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        poly = _largest_polygon(geom)
        if poly is None:
            continue
        simple = poly.simplify(0.000005, preserve_topology=True)
        coords = list(simple.exterior.coords)
        ring   = [[round(c[0], 6), round(c[1], 6)] for c in coords]
        if ring and ring[0] != ring[-1]:
            ring.append(ring[0])
        features.append({
            "type": "Feature",
            "properties": {"crown_index": int(i), "crown_label": f"crown_{i:04d}"},
            "geometry":   {"type": "Polygon", "coordinates": [ring]},
        })

    crowns_geojson = {"type": "FeatureCollection", "features": features}
    geojson_path   = viewer_dir / "crowns_wgs84.geojson"
    geojson_path.write_text(json.dumps(crowns_geojson))
    print(f"\nCrowns GeoJSON (WGS84): {geojson_path.name} ({len(features)} features)")

    # ── Manifest ──────────────────────────────────────────────────────────────
    manifest_blob = {
        "dataset":        dataset_name,
        "run_tag":        run_name,
        "num_oms":        int(num_oms),
        "num_crowns":     int(len(features)),
        "underlay_om_id": int(underlay_om_id),
        "crowns_geojson": geojson_path.name,
    }
    (viewer_dir / "manifest.json").write_text(json.dumps(manifest_blob, indent=2))

    # ── Phenology ─────────────────────────────────────────────────────────────
    print("\nLoading phenology data …")
    pheno_data = load_phenology_data(phenology_dir)

    # ── Build HTML ────────────────────────────────────────────────────────────
    html = build_html(
        dataset_name      = dataset_name,
        run_tag           = run_name,
        geojson_path_name = geojson_path.name,
        om_tile_manifest  = json.dumps(om_tile_entries),
        underlay_om_id    = underlay_om_id,
        geojson_inline    = json.dumps(crowns_geojson),
        manifest_inline   = json.dumps(manifest_blob),
        pheno_inline      = json.dumps(pheno_data),
        crs_epsg          = 4326,
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"\nWrote: {html_path}")

    # ── Phenology overview chart ──────────────────────────────────────────────
    scores_csv = Path(config.get("phenology_scores_csv",
                                  phenology_dir / "leafshed_tree_scores.csv"))
    phases_csv = phenology_dir / "leafshed_phenophase_by_om.csv"
    if scores_csv.exists() and phases_csv.exists():
        generate_phenology_overview(
            scores_csv, phases_csv, om_stems, viewer_dir / "phenology_overview.png")

    # ── Update config ─────────────────────────────────────────────────────────
    config["viewer_html"] = str(html_path)
    if "04b_interactive_viz" not in config.get("steps_completed", []):
        config["steps_completed"].append("04b_interactive_viz")
    save_config(config, config_path)
    print(f"Config updated: {config_path}")

    print(f"\nStep 4b complete.")
    print(f"Serve the viewer with:")
    print(f"  cd {viewer_dir}")
    print(f"  python -m http.server 8000")
    print(f"Then open: http://localhost:8000/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
