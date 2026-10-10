"""
FastAPI server for Phenology Pipeline Dashboard — v2.

New in v2:
  - Google sign-in (server-side token verification)          [auth.py]
  - PostgreSQL persistence of users and runs                 [db.py]
  - Per-user isolation: you only see and touch your own runs
  - File browser: list a run's folders and download any file [/files, /download]
  - Structured, human-readable error logging with tracebacks [errors.py]

Runs the bash pipeline on Linux/WSL/Docker, or the PowerShell one on Windows.
"""
import asyncio
import os
import re
import platform
import shutil
import signal
import subprocess
import threading
import uuid
import zipfile
import io
import json
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

import db
import auth
import errors
import airflow_client
import stac_utils
from log_config import get_logger

log = get_logger("server")

HERE = Path(__file__).parent.resolve()
# CoRE Stack §2: all data (inputs, outputs, logs) lives under /app/data.
# Runs keep their existing internal layout (uploads/, output/, pipeline.log,
# progress.json) — only the parent location moves out of the source tree.
# Overridable via DPM_RUNS_DIR for non-container / local dev use.
RUNS_DIR = Path(os.environ.get("DPM_RUNS_DIR", "/app/data/runs"))
TEMPLATES_DIR = HERE   # home.html lives alongside server.py

_ON_WINDOWS = platform.system() == "Windows"

# ---------------------------------------------------------------------------
# Locate pipeline script
# ---------------------------------------------------------------------------
PIPELINE_SCRIPT = None
_script_name = "run_pipeline.ps1" if _ON_WINDOWS else "run_pipeline.sh"
for candidate in [
    HERE.parent / "pipeline" / _script_name,
    (HERE.parent.parent.parent / "src" / "pipeline" / _script_name).resolve(),
]:
    if candidate.exists():
        PIPELINE_SCRIPT = candidate
        break
log.info("Platform: %s  Pipeline script: %s", platform.system(), PIPELINE_SCRIPT)

# ---------------------------------------------------------------------------
# Load .env for project root / model path
# ---------------------------------------------------------------------------
PROJECT_ROOT = None
MODEL_PATH = ""


def _win_to_wsl(p: str) -> str:
    m = re.match(r"^([A-Za-z]):\\?(.*)", p)
    if m:
        return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")
    return p


env_file = HERE / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if not _ON_WINDOWS:
            val = _win_to_wsl(val)
        if key == "DPM_PROJECT_ROOT":
            PROJECT_ROOT = Path(val).resolve()
        elif key == "DPM_MODEL_PATH":
            MODEL_PATH = val
log.info("PROJECT_ROOT: %s  MODEL_PATH: %s", PROJECT_ROOT, MODEL_PATH)

# Hard limit for one pipeline run triggered through /api/export-phenology.
# Kept just under the 7200 s the Airflow/STACD task waits for the HTTP
# response, so our own "timeout" error body arrives before Airflow gives up.
PIPELINE_TIMEOUT_S = int(os.environ.get("DPM_PIPELINE_TIMEOUT", "7000"))

# Max size (bytes) for a single orthomosaic upload.  Default: 2 GB.  Set 0 to disable.
MAX_OM_BYTES = int(os.environ.get("DPM_MAX_OM_BYTES", str(2 * 1024 ** 3)))

# ---------------------------------------------------------------------------
# API base URL — exposed to the frontend via /api/config so JS never
# hardcodes paths.  Set DPM_API_BASE_URL in .env:
#   /api            → same-origin, same container (default)
#   https://dpm.example.com/api  → cross-origin / CDN-hosted frontend
# ---------------------------------------------------------------------------
API_BASE_URL = os.environ.get("DPM_API_BASE_URL", "/api").rstrip("/")


def _resolve_model_path(params: dict) -> str:
    """
    Turn the UI's `model_type` (e.g. "urban_canopy") into a weights file.

    Looks for <model_type>.pth in, in order: $DPM_MODELS_DIR, the folder that
    holds the default model (DPM_MODEL_PATH), <project>/models and
    <project>/input/detectree_models. If nothing matches, the default
    DPM_MODEL_PATH is used — i.e. exactly the previous behaviour.
    """
    model_type = str((params or {}).get("model_type") or "").strip()
    # Plain file stem only — no slashes, so it cannot point outside the folders.
    if model_type and re.fullmatch(r"[A-Za-z0-9_.-]+", model_type):
        search_dirs: list[Path] = []
        if os.environ.get("DPM_MODELS_DIR"):
            search_dirs.append(Path(os.environ["DPM_MODELS_DIR"]))
        if MODEL_PATH:
            search_dirs.append(Path(MODEL_PATH).parent)
        if PROJECT_ROOT:
            search_dirs.append(PROJECT_ROOT / "models")
            search_dirs.append(PROJECT_ROOT / "input" / "detectree_models")
        search_dirs = list(dict.fromkeys(search_dirs))   # drop duplicates, keep order
        for d in search_dirs:
            candidate = d / f"{model_type}.pth"
            if candidate.is_file():
                return str(candidate)
        log.warning("model_type=%s: no %s.pth in %s — using default model %s",
                    model_type, model_type,
                    [str(d) for d in search_dirs], MODEL_PATH or "(pipeline default)")
    return str(MODEL_PATH) if MODEL_PATH else ""

STEP_KEYS = [
    "00_discover_oms", "01_crown_detection", "02_crown_tracking",
    "03_phenology_analysis", "03b_phenophase_classification",
    "04a_cog_tiling", "04b_interactive_viz",
]

# Fraction of the whole run each step is expected to take. Used to turn the
# per-step "cur/tot" PROGRESS markers into one smooth 0..1 global bar.
# Crown detection (GPU inference per OM) dominates, so it gets the biggest slice.
STEP_WEIGHTS = {
    "00_discover_oms":               0.02,
    "01_crown_detection":            0.50,
    "02_crown_tracking":             0.12,
    "03_phenology_analysis":         0.14,
    "03b_phenophase_classification": 0.04,
    "04a_cog_tiling":                0.13,
    "04b_interactive_viz":           0.05,
}
# Cumulative start offset of each step within the 0..1 bar.
_STEP_START: dict[str, float] = {}
_acc = 0.0
for _k in STEP_KEYS:
    _STEP_START[_k] = _acc
    _acc += STEP_WEIGHTS.get(_k, 0.0)

# Human-readable names for the progress-bar label.
STEP_LABELS = {
    "00_discover_oms":               "Discovering orthomosaics",
    "01_crown_detection":            "Detecting crowns",
    "02_crown_tracking":             "Tracking crowns across dates",
    "03_phenology_analysis":         "Computing phenology",
    "03b_phenophase_classification": "Classifying phenophase",
    "04a_cog_tiling":                "Generating tiles",
    "04b_interactive_viz":           "Building viewer",
}


def _progress_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "progress.json"


def _read_progress(run_id: str) -> dict:
    """Read the per-run progress.json written by the pipeline loop (best effort)."""
    p = _progress_path(run_id)
    if not p.exists():
        return {}
    try:
        import json as _json
        return _json.loads(p.read_text())
    except Exception:                                # noqa: BLE001
        return {}


def _write_progress(run_id: str, step: str, progress: float,
                    cur: int, tot: int, label: str) -> None:
    """Write the fine-grained progress detail for a run (best effort)."""
    try:
        import json as _json
        payload = {
            "step":         step,
            "step_label":   STEP_LABELS.get(step, step),
            "step_progress": round(float(progress), 4),
            "om_current":   int(cur),
            "om_total":     int(tot),
            "detail":       label,
        }
        _progress_path(run_id).write_text(_json.dumps(payload))
    except Exception:                                # noqa: BLE001
        pass

app = FastAPI()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def _startup():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    TEMPLATES_DIR.mkdir(exist_ok=True)
    log.info("Connecting to database ...")
    await db.connect()
    await db.init_db()
    log.info("Server ready - listening on 0.0.0.0:8000")


@app.on_event("shutdown")
async def _shutdown():
    log.info("Shutting down - closing DB pool")
    await db.disconnect()


# ---------------------------------------------------------------------------
# Health check — lightweight, no DB, no auth (used by the Docker healthcheck
# and by the cluster checklist). Referenced in docker-compose.yml.
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _uploads_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id / "uploads"

def _output_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id / "output"

def _log_path(run_id: str) -> Path:
    return RUNS_DIR / run_id / "pipeline.log"


def _thumb_path(run_id: str) -> Path:
    return _output_dir(run_id) / "04_viewer" / "thumbnail.jpg"


def _thumb_url(run_id: str) -> str | None:
    """Return the public thumbnail URL if thumbnail.jpg exists for this run."""
    return f"/viewer/{run_id}/thumbnail.jpg" if _thumb_path(run_id).exists() else None


def _generate_thumbnail(run_id: str) -> None:
    """
    Read the underlay COG for this run, downsample to ≤300 px wide using
    rasterio's built-in overview levels (fast, no resampling from full res),
    and write a compressed JPEG to 04_viewer/thumbnail.jpg.

    Uses only rasterio + numpy — both present in the dpm-tracking conda env.
    Safe to call from a background thread (blocking I/O only, no event loop).
    Silently skips if the COG or manifest are missing.
    """
    try:
        import numpy as _np
        import rasterio as _rio
        from rasterio.enums import Resampling as _RS

        viewer   = _output_dir(run_id) / "04_viewer"
        manifest = viewer / "tile_manifest.json"
        if not manifest.exists():
            return

        m   = json.loads(manifest.read_text())
        oms = m.get("oms") or []
        if not oms:
            return

        underlay_id = m.get("underlay_om_id")
        entry = next((o for o in oms if o.get("om_id") == underlay_id), oms[0])
        stem  = entry.get("stem")
        om_id = entry.get("om_id")
        if stem is None or om_id is None:
            return

        cog_path = viewer / "cogs" / f"OM{int(om_id):02d}_{stem}.tif"
        if not cog_path.exists():
            return

        target_w = 300  # max thumbnail width in pixels

        with _rio.open(str(cog_path)) as src:
            full_w, full_h = src.width, src.height
            # pick the overview level closest to target_w (or just read full
            # res at low scale if no overviews)
            scale = max(1, full_w // target_w)
            out_w = full_w  // scale
            out_h = full_h // scale

            # read the first 3 bands (or fewer if single-band)
            band_count = min(src.count, 3)
            bands = list(range(1, band_count + 1))
            data = src.read(
                bands,
                out_shape=(band_count, out_h, out_w),
                resampling=_RS.average,
            )   # shape: (C, H, W), dtype varies

        # normalise to uint8 per band, clipping to 2nd–98th percentile
        rgb = _np.zeros((out_h, out_w, 3), dtype=_np.uint8)
        for i in range(band_count):
            b = data[i].astype(_np.float32)
            lo, hi = _np.percentile(b[b > 0], [2, 98]) if (b > 0).any() else (b.min(), b.max())
            if hi > lo:
                b = (b - lo) / (hi - lo) * 255.0
            b = _np.clip(b, 0, 255).astype(_np.uint8)
            rgb[:, :, i] = b
        if band_count == 1:
            rgb[:, :, 1] = rgb[:, :, 0]
            rgb[:, :, 2] = rgb[:, :, 0]

        # write JPEG using rasterio (no Pillow needed)
        out_path = _thumb_path(run_id)
        with _rio.open(
            str(out_path), "w",
            driver="JPEG",
            height=out_h, width=out_w,
            count=3, dtype=_np.uint8,
        ) as dst:
            for i in range(3):
                dst.write(rgb[:, :, i], i + 1)

        log.info("Thumbnail written  run=%s  path=%s  size=%dx%d",
                 run_id, out_path.name, out_w, out_h)
    except Exception as exc:                           # noqa: BLE001
        log.warning("Thumbnail generation skipped  run=%s  reason=%s", run_id, exc)


# ---------------------------------------------------------------------------
# Static pages
# ---------------------------------------------------------------------------
def _find_page(name: str) -> Path | None:
    for p in (TEMPLATES_DIR / name, HERE / name):
        if p.exists():
            return p
    return None


@app.get("/")
def landing():
    """Public landing page (index.html). Shown to all visitors, logged in or not."""
    p = _find_page("index.html")
    if not p:
        raise HTTPException(404, "index.html not found")
    return FileResponse(str(p))


@app.get("/dashboard")
def dashboard():
    """Dashboard (home.html). Auth is handled client-side via the stored session token."""
    p = _find_page("home.html")
    if not p:
        raise HTTPException(404, "home.html not found")
    return FileResponse(str(p))


@app.get("/walkthrough")
def walkthrough_page():
    p = _find_page("walkthrough.html")
    if not p:
        raise HTTPException(404, "walkthrough.html not found")
    return FileResponse(str(p))


@app.get("/parameter-guide")
def parameter_guide_page():
    p = _find_page("parameter-guide.html")
    if not p:
        raise HTTPException(404, "parameter-guide.html not found")
    return FileResponse(str(p))


@app.get("/drone-guide")
def drone_guide_page():
    p = _find_page("drone-guide.html")
    if not p:
        raise HTTPException(404, "drone-guide.html not found")
    return FileResponse(str(p))


@app.get("/new-run")
def new_run_page():
    """Standalone new-run page (new-run.html). Reached from the dashboard's
    'New run' button. Like the other pages, auth is handled client-side."""
    p = _find_page("new-run.html")
    if not p:
        raise HTTPException(404, "new-run.html not found")
    return FileResponse(str(p))


@app.get("/param-docs.js")
def param_docs_js():
    p = _find_page("param-docs.js")
    if not p:
        raise HTTPException(404, "param-docs.js not found")
    return FileResponse(str(p), media_type="application/javascript")


@app.get("/api/config")
def get_config():
    """Public config the frontend needs to bootstrap sign-in and API calls."""
    return {
        "auth_enabled": auth.AUTH_ENABLED,
        "google_client_id": auth.GOOGLE_CLIENT_ID,
        "max_om_bytes": MAX_OM_BYTES,
        "api_base_url": API_BASE_URL,
    }


@app.get("/api/public/runs")
async def public_runs():
    """
    Finished runs whose owners have marked them public. No auth — used by the
    landing page (index.html) to list example analyses. Only runs with a
    generated viewer are returned.
    """
    runs = await db.list_public_runs()
    result = []
    for run in runs:
        viewer_index = _output_dir(run["id"]) / "04_viewer" / "index.html"
        if not viewer_index.exists():
            continue
        result.append({
            "id":         run["id"],
            "name":       run["run_name"],
            "num_orthos": run.get("num_orthos", 0),
            "created_at": run.get("created_at", ""),
            "viewer_url": f"/viewer/{run['id']}/index.html",
            "thumb_url":  _thumb_url(run["id"]),
        })
    return result


@app.get("/api/debug/thumb/{run_id}")
async def debug_thumb(run_id: str, user: dict = Depends(auth.get_current_user)):
    """
    Debug helper — shows thumbnail state and (re)generates it on demand.
    Remove once thumbnail behaviour is confirmed working.
    Call: GET /api/debug/thumb/<run_id>
    """
    viewer       = _output_dir(run_id) / "04_viewer"
    manifest_path = viewer / "tile_manifest.json"
    thumb        = _thumb_path(run_id)
    info: dict   = {
        "viewer_dir":       str(viewer),
        "manifest_exists":  manifest_path.exists(),
        "cogs_dir":         str(viewer / "cogs"),
        "cogs":             [p.name for p in (viewer / "cogs").iterdir()] if (viewer / "cogs").is_dir() else [],
        "thumbnail_exists": thumb.exists(),
        "thumbnail_path":   str(thumb),
        "thumb_url":        None,
        "manifest_oms":     [],
    }
    if manifest_path.exists():
        try:
            m = json.loads(manifest_path.read_text())
            info["manifest_oms"] = [
                {"om_id": o.get("om_id"), "stem": o.get("stem")}
                for o in (m.get("oms") or [])
            ]
            info["underlay_om_id"] = m.get("underlay_om_id")
        except Exception as e:
            info["manifest_parse_error"] = str(e)
    # regenerate if missing
    if not thumb.exists():
        _generate_thumbnail(run_id)
    info["thumbnail_exists_after"] = thumb.exists()
    info["thumb_url"] = _thumb_url(run_id)
    return info


@app.post("/api/runs/{run_id}/generate-thumbnail")
async def generate_thumbnail_endpoint(run_id: str, user: dict = Depends(auth.get_current_user)):
    """
    (Re)generate thumbnail.jpg for an existing completed run.
    Useful for backfilling thumbnails on runs that finished before this feature
    was added. Owner-only; runs synchronously (small image, fast).
    """
    await auth.require_run_owner(run_id, user)
    _generate_thumbnail(run_id)
    url = _thumb_url(run_id)
    if url is None:
        raise HTTPException(500, "Thumbnail generation failed — check server logs for details")
    return {"thumb_url": url}


@app.get("/api/me")
async def whoami(user: dict = Depends(auth.get_current_user)):
    return user


# ---------------------------------------------------------------------------
# Auth endpoint — Google SSO only. Returns a session token the browser stores
# and sends on every subsequent request as Bearer <token>.
# ---------------------------------------------------------------------------
@app.post("/api/auth/google")
async def google_login(req: auth.GoogleReq):
    claims = auth.verify_google_token(req.credential)
    email = claims["email"].strip().lower()
    name = claims.get("name")
    picture = claims.get("picture")
    await db.upsert_user(email, name, picture, auth_provider="google")
    token = auth.issue_session_token(email, name)
    return {"token": token, "user": {"email": email, "name": name, "picture": picture}}


# ---------------------------------------------------------------------------
# Runs (all scoped to the authenticated user)
# ---------------------------------------------------------------------------
class CreateRunReq(BaseModel):
    run_name: str
    is_public: bool = True


@app.post("/api/runs", status_code=201)
async def create_run(req: CreateRunReq, user: dict = Depends(auth.get_current_user)):
    run_id = str(uuid.uuid4())
    _uploads_dir(run_id).mkdir(parents=True, exist_ok=True)
    _output_dir(run_id).mkdir(parents=True, exist_ok=True)
    await db.create_run(run_id, user["email"], req.run_name, req.is_public)
    return {"id": run_id}


def _attach_progress(run: dict) -> dict:
    """Merge fine-grained progress detail (from progress.json) into a run dict."""
    prog = _read_progress(run["id"])
    if prog:
        run["om_current"] = prog.get("om_current", 0)
        run["om_total"]   = prog.get("om_total", 0)
        run["step_detail"] = prog.get("detail", "")
        run["step_label"]  = prog.get("step_label", "")
        # progress.json is written more often than the DB column, so prefer it
        # while the run is still active.
        if run.get("status") == "running" and prog.get("step_progress") is not None:
            run["step_progress"] = prog["step_progress"]
    return run


@app.get("/api/runs")
async def list_runs(user: dict = Depends(auth.get_current_user)):
    runs = await db.list_runs_for_user(user["email"])
    for run in runs:
        viewer_index = _output_dir(run["id"]) / "04_viewer" / "index.html"
        run["has_viewer"] = viewer_index.exists()
        _attach_progress(run)
    return runs


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str, user: dict = Depends(auth.get_current_user)):
    run = await auth.require_run_owner(run_id, user)
    viewer_index = _output_dir(run_id) / "04_viewer" / "index.html"
    run["has_viewer"] = viewer_index.exists()
    _attach_progress(run)
    return run


@app.post("/api/runs/{run_id}/upload")
async def upload_file(run_id: str, file: UploadFile, user: dict = Depends(auth.get_current_user)):
    await auth.require_run_owner(run_id, user)
    dest = _uploads_dir(run_id) / file.filename
    content = await file.read()
    # ── Size guard ──────────────────────────────────────────────────────────
    if MAX_OM_BYTES and len(content) > MAX_OM_BYTES:
        mb_limit = MAX_OM_BYTES / 1024 ** 2
        mb_actual = len(content) / 1024 ** 2
        raise HTTPException(
            413,
            f"File too large: {file.filename} is {mb_actual:.0f} MB "
            f"(limit is {mb_limit:.0f} MB per orthomosaic). "
            "Compress or tile the orthomosaic before uploading."
        )
    # ────────────────────────────────────────────────────────────────────────
    with open(dest, "wb") as f:
        f.write(content)

    # Record in orthos table
    current_count = await db.count_orthos(run_id)
    await db.create_ortho(
        run_id=run_id,
        original_filename=file.filename,
        size_bytes=len(content),
        upload_order=current_count + 1,
    )
    # Update run totals
    num = await db.count_orthos(run_id)
    total = await db.sum_ortho_bytes(run_id)
    await db.update_run(run_id, num_orthos=num, total_bytes=total)

    return {"filename": file.filename, "size": len(content)}


@app.post("/api/runs/{run_id}/extract-dates")
async def extract_dates(run_id: str, user: dict = Depends(auth.get_current_user)):
    """Legacy endpoint — kept for backward compatibility. Returns all files as needing dates."""
    await auth.require_run_owner(run_id, user)
    uploads = _uploads_dir(run_id)
    tifs = sorted(uploads.glob("*.tif")) + sorted(uploads.glob("*.tiff"))
    if not tifs:
        raise HTTPException(400, "No .tif files uploaded")
    missing = {tif.name: None for tif in tifs}
    return {"dates_found": {}, "missing_dates_map": missing, "all_have_dates": False}


class ApplyDatesReq(BaseModel):
    date_map: dict   # {"original_filename.tif": "DD/MM/YYYY", ...}


@app.post("/api/runs/{run_id}/apply-dates")
async def apply_dates(run_id: str, req: ApplyDatesReq, user: dict = Depends(auth.get_current_user)):
    """
    Rename uploaded TIFs internally for the pipeline using the user-provided dates.

    The pipeline expects filenames like  sit_DD-MM-YY.tif  (chronological sort key).
    The user's original filename is stored in a mapping JSON so the UI can still show
    the original name.  The user never sees the renamed files.

    Renames: original.tif  →  sit_DD-MM-YY.tif  (sorted chronologically and numbered
    to avoid collisions when two OMs share the same date).
    """
    await auth.require_run_owner(run_id, user)
    uploads = _uploads_dir(run_id)
    from datetime import datetime as dt

    # Parse and sort by date
    entries = []
    for original_name, date_str in req.date_map.items():
        src = uploads / original_name
        if not src.exists():
            raise HTTPException(400, f"File not found: {original_name}")
        try:
            d = dt.strptime(date_str.strip(), "%d/%m/%Y")
        except ValueError as e:
            raise HTTPException(400, f"Invalid date '{date_str}' for {original_name}: {e}")
        entries.append({"original": original_name, "date": d, "path": src})

    entries.sort(key=lambda e: e["date"])

    # Rename: sit_DD-MM-YY.tif  (add _dateNotConfirmed if needed to match pipeline expectations)
    mapping = {}   # original_name → pipeline_name
    seen_dates = {}
    for entry in entries:
        d = entry["date"]
        date_tag = d.strftime("%d-%m-%y")
        # Handle duplicate dates by appending a counter
        count = seen_dates.get(date_tag, 0)
        seen_dates[date_tag] = count + 1
        suffix = f"_{count}" if count > 0 else ""
        pipeline_name = f"sit_{date_tag}{suffix}.tif"

        src = entry["path"]
        dst = uploads / pipeline_name
        if src.name != pipeline_name:
            # Avoid overwriting: if dst exists and is a different file, add counter
            while dst.exists() and dst != src:
                count += 1
                seen_dates[date_tag] = count
                pipeline_name = f"sit_{date_tag}_{count}.tif"
                dst = uploads / pipeline_name
            src.rename(dst)

        mapping[entry["original"]] = pipeline_name

    # Save the mapping so the UI can display original names
    mapping_file = _output_dir(run_id) / "filename_mapping.json"
    mapping_file.parent.mkdir(parents=True, exist_ok=True)
    import json as _json
    mapping_file.write_text(_json.dumps(mapping, indent=2))

    # Update orthos table with pipeline filenames and acquisition dates
    for entry in entries:
        ortho = await db.get_ortho_by_filename(run_id, entry["original"])
        if ortho:
            await db.update_ortho(
                ortho["id"],
                pipeline_filename=mapping[entry["original"]],
                acquisition_date=entry["date"].date(),
            )

    return {"renamed": mapping}


class RenameDatesReq(BaseModel):
    date_overrides: dict


@app.post("/api/runs/{run_id}/rename-dates")
async def rename_dates(run_id: str, req: RenameDatesReq, user: dict = Depends(auth.get_current_user)):
    """Legacy endpoint — redirects to apply-dates logic."""
    await auth.require_run_owner(run_id, user)
    uploads = _uploads_dir(run_id)
    renamed = {}
    from datetime import datetime as dt
    for original_name, date_str in req.date_overrides.items():
        src = uploads / original_name
        if not src.exists():
            raise HTTPException(400, f"File not found: {original_name}")
        try:
            d = dt.strptime(date_str.strip(), "%d/%m/%Y")
            suffix = d.strftime("%d-%m-%y")
            new_path = uploads / f"sit_{suffix}{src.suffix}"
            if new_path.exists() and new_path != src:
                new_path = uploads / f"sit_{suffix}_1{src.suffix}"
            src.rename(new_path)
            renamed[original_name] = new_path.name
        except ValueError as e:
            raise HTTPException(400, f"Invalid date for {original_name}: {e}")
    return {"renamed": renamed}


@app.patch("/api/runs/{run_id}/params")
async def update_params(run_id: str, body: dict, user: dict = Depends(auth.get_current_user)):
    run = await auth.require_run_owner(run_id, user)
    if run["status"] == "running":
        raise HTTPException(409, "Cannot update params while running")
    merged = {**(run.get("params") or {}), **body}
    await db.update_run(run_id, params=merged)
    return {"updated": True}


class VisibilityReq(BaseModel):
    is_public: bool


@app.patch("/api/runs/{run_id}/visibility")
async def set_visibility(run_id: str, req: VisibilityReq,
                         user: dict = Depends(auth.get_current_user)):
    """Toggle whether this run appears on the public landing page (owner only)."""
    await auth.require_run_owner(run_id, user)
    await db.update_run(run_id, is_public=req.is_public)
    return {"is_public": req.is_public}


@app.post("/api/runs/{run_id}/start")
async def start_run(run_id: str, background_tasks: BackgroundTasks,
                    user: dict = Depends(auth.get_current_user)):
    run = await auth.require_run_owner(run_id, user)
    params = run.get("params") or {}
    log.info("Compute trigger  run=%s  user=%s  airflow=%s",
             run_id, user["email"], airflow_client.AIRFLOW_ENABLED)

    if airflow_client.AIRFLOW_ENABLED:
        # ── Airflow path (§2/§8): trigger the DAG and record the dag_run_id ──
        conf = {"run_id": run_id, **params}
        try:
            dag_run_id = await airflow_client.trigger_conf(conf)
        except airflow_client.AirflowClientError as e:
            log.error("Airflow trigger failed  run=%s  error=%s", run_id, e)
            raise HTTPException(502, f"Airflow trigger failed: {e}")
        log.info("Airflow triggered  run=%s  dag_run_id=%s", run_id, dag_run_id)
        await db.update_run(run_id, status="running",
                            started_at=datetime.now(timezone.utc))
        # Record dag_run_id so /api/dag/status can poll it.
        # Store in the run's params (no schema change needed).
        merged_params = {**params, "_dag_run_id": dag_run_id}
        await db.update_run(run_id, params=merged_params)
        # Create a job record for this compute attempt
        job_id = str(uuid.uuid4())
        await db.create_job(run_id, job_type="airflow",
                            request_id=dag_run_id)
        return {"status": "started", "run_id": run_id,
                "mode": "airflow", "dag_run_id": dag_run_id}
    else:
        # ── Local path: run inside this container ──
        if not PIPELINE_SCRIPT:
            log.error("Pipeline script not found on server")
            raise HTTPException(500, "Pipeline script not found on server")
        background_tasks.add_task(_run_pipeline, run_id, user["email"], params)
        return {"status": "started", "run_id": run_id, "mode": "local"}


# ---------------------------------------------------------------------------
# Airflow proxy endpoints (§8: UI → your /api/dag/run and /api/dag/status.
# Never call Airflow from JavaScript — no CORS; credentials stay off the page.)
# ---------------------------------------------------------------------------
class DagRunReq(BaseModel):
    run_id: str
    params: dict = {}


@app.post("/api/dag/run")
async def dag_run(req: DagRunReq, user: dict = Depends(auth.get_current_user)):
    """
    Proxy: trigger the Airflow DAG from the UI.
    This is the endpoint the frontend calls instead of hitting Airflow directly.
    """
    if not airflow_client.AIRFLOW_ENABLED:
        raise HTTPException(400, "Airflow is not configured — AIRFLOW_API_BASE is empty")
    # Verify the user owns this run
    await auth.require_run_owner(req.run_id, user)
    conf = {"run_id": req.run_id, **req.params}
    try:
        dag_run_id = await airflow_client.trigger_conf(conf)
    except airflow_client.AirflowClientError as e:
        log.error("Airflow trigger failed  run=%s  error=%s", req.run_id, e)
        raise HTTPException(502, f"Airflow trigger failed: {e}")
    log.info("DAG triggered via proxy  run=%s  dag_run_id=%s  user=%s",
             req.run_id, dag_run_id, user["email"])
    return {"dag_run_id": dag_run_id, "status": "triggered"}


@app.get("/api/dag/status")
async def dag_status(dag_run_id: str, user: dict = Depends(auth.get_current_user)):
    """
    Proxy: poll the Airflow DAG run status.
    The frontend polls this endpoint — never Airflow directly.
    """
    if not airflow_client.AIRFLOW_ENABLED:
        raise HTTPException(400, "Airflow is not configured — AIRFLOW_API_BASE is empty")
    try:
        state = await airflow_client.run_state(dag_run_id)
    except airflow_client.AirflowClientError as e:
        log.error("Airflow status poll failed  dag_run_id=%s  error=%s", dag_run_id, e)
        raise HTTPException(502, f"Airflow status poll failed: {e}")
    return state


# ---------------------------------------------------------------------------
# Blocking pipeline runner for the Airflow callback.
#
# /api/export-phenology must keep the HTTP request open until the pipeline is
# done (Airflow is waiting on it), but it must NOT run the pipeline on the
# event loop — that would freeze every other request, /health included, for
# the whole run. So the callback awaits this function in a worker thread
# (asyncio.to_thread). It touches only files, never the DB pool, so it is
# safe off the event loop.
# ---------------------------------------------------------------------------
def _marker_progress(line: str):
    """
    Parse one line of pipeline output.
    Returns (step_key, global_progress, cur, tot, label) for a STEP:/PROGRESS:
    marker, or None for an ordinary line.
    """
    stripped = line.strip()
    if stripped.startswith("STEP:"):
        key = stripped[5:].strip()
        if key in STEP_KEYS:
            return key, round(_STEP_START.get(key, 0.0), 4), 0, 0, STEP_LABELS.get(key, key)
        return None
    if stripped.startswith("PROGRESS:"):
        # PROGRESS:<step_key>:<cur>/<tot>:<label>
        parts = stripped[len("PROGRESS:"):].split(":", 2)
        if len(parts) >= 2 and parts[0] in STEP_WEIGHTS:
            step = parts[0]
            label = parts[2].strip() if len(parts) >= 3 else STEP_LABELS.get(step, step)
            cur = tot = 0
            within = 0.0
            cs, sep, ts = parts[1].strip().partition("/")
            if sep:
                try:
                    cur, tot = int(cs), int(ts)
                    within = (cur / tot) if tot > 0 else 0.0
                except ValueError:
                    cur = tot = 0
            within = max(0.0, min(1.0, within))
            progress = round(_STEP_START.get(step, 0.0)
                             + within * STEP_WEIGHTS.get(step, 0.0), 4)
            return step, progress, cur, tot, label
    return None


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill the pipeline AND its children (bash -> conda run -> python)."""
    try:
        if _ON_WINDOWS:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True)
        else:
            # start_new_session=True below makes proc.pid the process-group id.
            os.killpg(proc.pid, signal.SIGKILL)
    except Exception:                                # noqa: BLE001
        try:
            proc.kill()
        except Exception:                            # noqa: BLE001
            pass


def _run_pipeline_streaming(run_id: str, cmd: list, timeout_s: float):
    """
    Run the pipeline to completion. BLOCKING — call via asyncio.to_thread.

    Streams output line by line into pipeline.log (so the log is readable
    while the run is in flight) and keeps progress.json current from the
    STEP:/PROGRESS: markers (so the dashboard progress bar moves).

    Returns (returncode, full_output, timed_out).
    """
    run_log = _log_path(run_id)
    run_log.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []
    timed_out = threading.Event()

    with open(run_log, "w", buffering=1, encoding="utf-8") as lf:
        lf.write(f"[server] cmd  : {' '.join(str(a) for a in cmd)}\n")
        lf.write(f"[server] cwd  : {PIPELINE_SCRIPT.parent}\n")
        lf.write("[server] mode : airflow callback (/api/export-phenology)\n\n")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
            bufsize=1, cwd=str(PIPELINE_SCRIPT.parent),
            start_new_session=not _ON_WINDOWS,
        )

        def _on_timeout():
            timed_out.set()
            _kill_process_tree(proc)

        timer = threading.Timer(timeout_s, _on_timeout)
        timer.daemon = True
        timer.start()
        try:
            for line in proc.stdout:
                lf.write(line)
                parts.append(line)
                marker = _marker_progress(line)
                if marker:
                    step, progress, cur, tot, label = marker
                    _write_progress(run_id, step, progress, cur, tot, label)
            proc.wait()
        finally:
            timer.cancel()

        if timed_out.is_set():
            lf.write(f"\n[server] pipeline killed after {timeout_s:.0f}s timeout\n")

    return proc.returncode, "".join(parts), timed_out.is_set()


# ---------------------------------------------------------------------------
# Airflow callback — POST /api/export-phenology  (§7/§8)
#
# This is what the Airflow DAG calls back to (via CORESTACK_API_BASE) to run
# the pipeline and collect the STAC result. The algorithm repo YAML points:
#   url: "http://dpm-dashboard:8000/api/export-phenology"
#
# STACD §12 HTTP status contract:
#   200  success  → asset registered
#   400  skipped  → invalid params
#   404  skipped  → no data for this run
#   500  failed   → pipeline/computation error
# ---------------------------------------------------------------------------
class ExportPhenologyReq(BaseModel):
    run_id: str
    # All other pipeline params are optional — they override the run's stored params
    model_type: str | None = None
    tile_width: int | None = None
    tile_height: int | None = None
    tile_buffer: int | None = None
    fixed_iou: float | None = None
    base_threshold_tag: str | None = None
    align_threshold_tag: str | None = None
    align_method: str | None = None
    w_veg_amp: float | None = None
    w_depth: float | None = None
    w_gcc_amp: float | None = None
    w_tex: float | None = None
    exclude_stems: str | None = None
    skip_viz: bool | None = None


@app.post("/api/export-phenology")
async def export_phenology(req: ExportPhenologyReq):
    """
    Airflow callback endpoint. The DAG's API-mode task POSTs the run conf here.
    Runs the pipeline in a worker thread and waits for it (Airflow is holding the
    request open), builds a STAC Item from the output, and returns the STACD
    response envelope. The event loop stays free the whole time.

    No user auth — this is called from the Airflow worker, not a browser.
    The worker authenticates via the AIRFLOW_TOKEN / basic auth it already uses
    to talk to Airflow's own API. For additional security, consider adding a
    bearer token check (see areas/api_security.md).
    """
    run_id = req.run_id
    if not run_id:
        return JSONResponse(
            status_code=400,
            content=stac_utils.build_error_response("missing_param", "run_id is required"),
        )

    # Look up the run in the DB
    run = await db.get_run(run_id)
    if run is None:
        return JSONResponse(
            status_code=404,
            content=stac_utils.build_error_response("not_found", f"Run {run_id} not found"),
        )

    # Merge request params over stored run params
    stored_params = run.get("params") or {}
    override = {k: v for k, v in req.model_dump(exclude={"run_id"}).items() if v is not None}
    params = {**stored_params, **override}

    log.info("Export-phenology callback  run=%s  params=%s", run_id, list(params.keys()))

    # Check that orthomosaics exist
    om_dir = _uploads_dir(run_id)
    tifs = list(om_dir.glob("*.tif")) + list(om_dir.glob("*.tiff"))
    if not tifs:
        log.warning("Export-phenology: no orthomosaics  run=%s", run_id)
        return JSONResponse(
            status_code=404,
            content=stac_utils.build_error_response("no_data", f"No orthomosaics found for run {run_id}"),
        )

    # Run the pipeline synchronously (Airflow is waiting)
    if not PIPELINE_SCRIPT:
        log.error("Pipeline script not found")
        return JSONResponse(
            status_code=500,
            content=stac_utils.build_error_response("server_error", "Pipeline script not found on server"),
        )

    out_dir = str(_output_dir(run_id))
    run_name = run.get("run_name") or run_id
    cmd = _build_cmd(run_name, str(om_dir), out_dir, params)
    log.info("Export-phenology: running pipeline  run=%s  timeout=%ss",
             run_id, PIPELINE_TIMEOUT_S)

    await db.update_run(run_id, status="running", step_progress=0.0,
                        started_at=datetime.now(timezone.utc))

    # The pipeline runs for minutes to hours. It runs in a worker thread, so
    # this coroutine only *waits* here: the event loop stays free to serve
    # /health, the dashboard and status polling while Airflow holds this
    # request open.
    try:
        returncode, raw_output, timed_out = await asyncio.to_thread(
            _run_pipeline_streaming, run_id, cmd, PIPELINE_TIMEOUT_S)
    except Exception as e:
        log.error("Export-phenology: pipeline exception  run=%s  error=%s",
                  run_id, e, exc_info=True)
        await db.update_run(run_id, status="failed",
                            error_msg=str(e),
                            finished_at=datetime.now(timezone.utc))
        return JSONResponse(
            status_code=500,
            content=stac_utils.build_error_response("pipeline_error", str(e)),
        )

    if timed_out:
        msg = f"Pipeline timed out ({PIPELINE_TIMEOUT_S}s limit)"
        log.error("Export-phenology: pipeline timed out  run=%s", run_id)
        await db.update_run(run_id, status="failed", error_msg=msg,
                            finished_at=datetime.now(timezone.utc))
        return JSONResponse(
            status_code=500,
            content=stac_utils.build_error_response("timeout", msg),
        )

    if returncode != 0:
        summary = errors.classify_pipeline_error(raw_output)
        log.error("Export-phenology: pipeline failed  run=%s  summary=%s", run_id, summary)
        await db.update_run(run_id, status="failed",
                            error_msg=summary,
                            finished_at=datetime.now(timezone.utc))
        return JSONResponse(
            status_code=500,
            content=stac_utils.build_error_response("pipeline_failed", summary),
        )

    # Pipeline succeeded — find the phenology GeoJSON output
    phenoclf = Path(out_dir) / "03_phenology" / "tree_master_geojson_phenoclf.geojson"
    if not phenoclf.exists():
        # Step 3b (phenophase classifier) did not run — fall back to the
        # step-3 file. The STAC Item says so (see phenophase_classified below).
        phenoclf = Path(out_dir) / "03_phenology" / "tree_master_geojson.geojson"
    if not phenoclf.exists():
        log.warning("Export-phenology: no output GeoJSON  run=%s", run_id)
        await db.update_run(run_id, status="failed",
                            error_msg="Pipeline succeeded but no phenology GeoJSON produced",
                            finished_at=datetime.now(timezone.utc))
        return JSONResponse(
            status_code=404,
            content=stac_utils.build_error_response("no_output", "Pipeline ran but produced no phenology GeoJSON"),
        )
    phenophase_classified = phenoclf.name.endswith("_phenoclf.geojson")
    if not phenophase_classified:
        log.warning("Export-phenology: step 3b output missing, exporting the "
                    "unclassified GeoJSON  run=%s", run_id)

    try:
        # File parsing and thumbnail rendering are blocking I/O too — keep
        # them off the event loop.
        geojson = await asyncio.to_thread(
            lambda: json.loads(phenoclf.read_text(encoding="utf-8")))

        # Build temporal coverage from ortho acquisition dates
        orthos = await db.list_orthos(run_id)
        dates = sorted([o["acquisition_date"] for o in orthos if o.get("acquisition_date")])
        start_dt = dates[0] + "T00:00:00Z" if dates else None
        end_dt = dates[-1] + "T00:00:00Z" if dates else None

        # Thumbnail
        await asyncio.to_thread(_generate_thumbnail, run_id)
        thumb_url = _thumb_url(run_id)

        # Construct the STAC Item
        corestack_base = os.environ.get("CORESTACK_API_BASE", "").rstrip("/")
        data_href = (
            f"{corestack_base}/dl/{run_id}/file?path=output/03_phenology/{phenoclf.name}"
            if corestack_base
            else f"/dl/{run_id}/file?path=output/03_phenology/{phenoclf.name}"
        )

        # Lineage: record the model that was actually loaded, not just the
        # name that was requested (the resolver falls back to the default
        # when <model_type>.pth is not on disk).
        lineage_params = dict(params)
        resolved_model = _resolve_model_path(params)
        if resolved_model:
            lineage_params["model_type"] = Path(resolved_model).stem

        stac_item = stac_utils.build_phenology_stac_item(
            run_id=run_id,
            run_name=run_name,
            geojson=geojson,
            data_href=data_href,
            start_datetime=start_dt,
            end_datetime=end_dt,
            thumbnail_href=thumb_url,
            params=lineage_params,
            phenophase_classified=phenophase_classified,
        )
    except Exception as e:
        # Without this the run would stay "running" forever if the export
        # step raised (e.g. unknown CRS, unreadable GeoJSON).
        log.error("Export-phenology: STAC export failed  run=%s  error=%s",
                  run_id, e, exc_info=True)
        await db.update_run(run_id, status="failed",
                            error_msg=f"STAC export failed: {e}",
                            finished_at=datetime.now(timezone.utc))
        return JSONResponse(
            status_code=500,
            content=stac_utils.build_error_response("stac_export_failed", str(e)),
        )

    await db.update_run(run_id, status="done", step_progress=1.0,
                        finished_at=datetime.now(timezone.utc))
    _write_progress(run_id, "04b_interactive_viz", 1.0, 0, 0, "Complete")
    log.info("Export-phenology: success  run=%s  features=%d",
             run_id, len(geojson.get("features", [])))

    return stac_utils.build_response_envelope(stac_item)


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: str, user: dict = Depends(auth.get_current_user)):
    await auth.require_run_owner(run_id, user)
    shutil.rmtree(RUNS_DIR / run_id, ignore_errors=True)
    await db.delete_run(run_id)  # CASCADE deletes orthos + jobs
    return {"deleted": run_id}


@app.get("/api/runs/{run_id}/log")
async def get_log(run_id: str, lines: int = 60, user: dict = Depends(auth.get_current_user)):
    await auth.require_run_owner(run_id, user)
    run_log = _log_path(run_id)
    if not run_log.exists():
        return {"lines": []}
    all_lines = run_log.read_text(errors="replace").splitlines()
    return {"lines": all_lines[-lines:]}


@app.get("/api/runs/{run_id}/error")
async def get_error(run_id: str, user: dict = Depends(auth.get_current_user)):
    """Return the structured error for a failed run (clean summary + traceback)."""
    run = await auth.require_run_owner(run_id, user)
    if run["status"] != "failed":
        return {"failed": False}
    run_log = _log_path(run_id)
    raw = run_log.read_text(errors="replace") if run_log.exists() else ""
    return {
        "failed": True,
        "summary": run.get("error_msg") or errors.classify_pipeline_error(raw),
        "traceback": errors.extract_python_traceback(raw),
        "step": run.get("current_step"),
    }


# ---------------------------------------------------------------------------
# Orthos (uploaded files for a run)
# ---------------------------------------------------------------------------
@app.get("/api/runs/{run_id}/orthos")
async def list_orthos(run_id: str, user: dict = Depends(auth.get_current_user)):
    """List all uploaded orthomosaics for a run with their original/pipeline names and dates."""
    await auth.require_run_owner(run_id, user)
    return await db.list_orthos(run_id)


@app.delete("/api/runs/{run_id}/orthos/{ortho_id}")
async def remove_ortho(run_id: str, ortho_id: str, user: dict = Depends(auth.get_current_user)):
    """Remove an ortho from a run (both DB record and file on disk)."""
    run = await auth.require_run_owner(run_id, user)
    if run["status"] == "running":
        raise HTTPException(409, "Cannot remove files while pipeline is running")
    orthos = await db.list_orthos(run_id)
    ortho = next((o for o in orthos if o["id"] == ortho_id), None)
    if not ortho:
        raise HTTPException(404, "Ortho not found")
    # Delete the file on disk (try both original and pipeline name)
    for fname in [ortho.get("pipeline_filename"), ortho["original_filename"]]:
        if fname:
            fpath = _uploads_dir(run_id) / fname
            if fpath.exists():
                fpath.unlink()
    await db.delete_ortho(ortho_id)
    # Update run totals
    num = await db.count_orthos(run_id)
    total = await db.sum_ortho_bytes(run_id)
    await db.update_run(run_id, num_orthos=num, total_bytes=total)
    return {"deleted": ortho_id}


# ---------------------------------------------------------------------------
# Jobs (compute history for a run)
# ---------------------------------------------------------------------------
@app.get("/api/runs/{run_id}/jobs")
async def list_jobs(run_id: str, user: dict = Depends(auth.get_current_user)):
    """List all compute attempts for a run (most recent first)."""
    await auth.require_run_owner(run_id, user)
    return await db.list_jobs(run_id)


@app.get("/api/runs/{run_id}/jobs/latest")
async def get_latest_job(run_id: str, user: dict = Depends(auth.get_current_user)):
    """Get the most recent job for a run."""
    await auth.require_run_owner(run_id, user)
    job = await db.get_latest_job(run_id)
    if not job:
        return {"job": None}
    return job


# ---------------------------------------------------------------------------
# Run stats (aggregated info for the dashboard)
# ---------------------------------------------------------------------------
@app.get("/api/runs/{run_id}/stats")
async def get_run_stats(run_id: str, user: dict = Depends(auth.get_current_user)):
    """Return aggregated stats for a run."""
    run = await auth.require_run_owner(run_id, user)
    orthos = await db.list_orthos(run_id)
    latest_job = await db.get_latest_job(run_id)
    return {
        "run_id": run_id,
        "status": run["status"],
        "num_orthos": len(orthos),
        "total_bytes": sum(o.get("size_bytes") or 0 for o in orthos),
        "dates_assigned": sum(1 for o in orthos if o.get("acquisition_date")),
        "date_range": {
            "earliest": min((o["acquisition_date"] for o in orthos if o.get("acquisition_date")), default=None),
            "latest": max((o["acquisition_date"] for o in orthos if o.get("acquisition_date")), default=None),
        },
        "latest_job": {
            "state": latest_job["state"] if latest_job else None,
            "stage": latest_job.get("current_stage") if latest_job else None,
            "progress": latest_job.get("progress", 0) if latest_job else 0,
        },
        "created_at": run.get("created_at"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
    }


# ---------------------------------------------------------------------------
# File browser  (list folders/files + download individual files or zip)
# ---------------------------------------------------------------------------
@app.get("/api/runs/{run_id}/files")
async def list_files(run_id: str, subpath: str = "", user: dict = Depends(auth.get_current_user)):
    """
    List the contents of a run's folder tree. `subpath` is relative to the run
    root (which contains 'uploads/' and 'output/'). Directory traversal is blocked.
    """
    await auth.require_run_owner(run_id, user)
    run_root = (RUNS_DIR / run_id).resolve()
    target = (run_root / subpath).resolve()

    # Block escaping the run directory (path-aware; not a raw string prefix)
    if not target.is_relative_to(run_root):
        raise HTTPException(400, "Invalid path")
    if not target.exists():
        raise HTTPException(404, "Path not found")
    if not target.is_dir():
        raise HTTPException(400, "Not a directory — use /download for files")

    entries = []
    for child in sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        rel = child.relative_to(run_root).as_posix()
        entries.append({
            "name": child.name,
            "path": rel,
            "is_dir": child.is_dir(),
            "size": child.stat().st_size if child.is_file() else None,
        })
    parent = "" if not subpath else str(Path(subpath).parent.as_posix())
    if parent == ".":
        parent = ""
    return {"cwd": subpath, "parent": parent, "entries": entries}


@app.get("/api/runs/{run_id}/download")
async def download_file(run_id: str, path: str, user: dict = Depends(auth.get_current_user)):
    """Download a single file from within the run folder."""
    await auth.require_run_owner(run_id, user)
    run_root = (RUNS_DIR / run_id).resolve()
    target = (run_root / path).resolve()
    if not target.is_relative_to(run_root):
        raise HTTPException(400, "Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(target), filename=target.name)


@app.get("/api/runs/{run_id}/download-all")
async def download_all(run_id: str, which: str = "output",
                       user: dict = Depends(auth.get_current_user)):
    """
    Stream a zip of the run's 'output' (default) or 'uploads' folder.
    which = output | uploads | all
    """
    await auth.require_run_owner(run_id, user)
    run_root = RUNS_DIR / run_id
    if which == "output":
        roots = [_output_dir(run_id)]
    elif which == "uploads":
        roots = [_uploads_dir(run_id)]
    else:
        roots = [_uploads_dir(run_id), _output_dir(run_id)]

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root in roots:
            if not root.exists():
                continue
            for f in root.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(run_root).as_posix())
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_{which}.zip"'},
    )



# ---------------------------------------------------------------------------
# Public download routes (no auth — UUID provides security)
# These work with browser <a download> links which can't send auth headers.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Crown-crop API — proxies TiTiler /cog/crop/ for exact bounding-box extraction
# ---------------------------------------------------------------------------
TITILER_BASE = "http://dpm-titiler:80"

@app.get("/api/runs/{run_id}/crop/{om_id}")
async def crown_crop(
    run_id: str, om_id: int,
    minx: float, miny: float, maxx: float, maxy: float,
    width: int = 256, height: int = 256,
):
    import json as _json
    width  = max(64, min(width, 1024))
    height = max(64, min(height, 1024))

    manifest_path = _output_dir(run_id) / "04_viewer" / "tile_manifest.json"
    if not manifest_path.exists():
        raise HTTPException(404, "Tile manifest not found")
    manifest = _json.loads(manifest_path.read_text())
    om_entry = next((o for o in manifest.get("oms", []) if o["om_id"] == om_id), None)
    if not om_entry:
        raise HTTPException(404, f"OM{om_id} not in manifest")

    cog_rel = om_entry.get("cog_url")
    if cog_rel:
        cog_path = (_output_dir(run_id) / cog_rel).resolve()
    else:
        stem = om_entry.get("stem", "")
        cog_path = (_output_dir(run_id) / "04_viewer" / "cogs" / f"OM{om_id:02d}_{stem}.tif").resolve()

    if not cog_path.exists():
        raise HTTPException(404, f"COG not found: {cog_path.name}")

    url = (
        f"{TITILER_BASE}/cog/bbox/{minx},{miny},{maxx},{maxy}.png"
        f"?url=file://{cog_path}"
        f"&width={width}&height={height}"
        f"&coord-crs=epsg:4326"
        f"&resampling=nearest"
    )

    import asyncio, urllib.request, urllib.error
    try:
        # Run blocking urllib call in a thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        def _fetch():
            # Bypass proxy for internal Docker hostnames
            proxy_handler = urllib.request.ProxyHandler({})
            opener = urllib.request.build_opener(proxy_handler)
            with opener.open(url, timeout=30) as resp:
                return resp.read()
        data = await loop.run_in_executor(None, _fetch)
        return StreamingResponse(
            iter([data]),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:200]
        raise HTTPException(502, f"TiTiler error {e.code}: {body}")
    except urllib.error.URLError as e:
        raise HTTPException(503, f"TiTiler not reachable: {e.reason}")
    except Exception as e:
        raise HTTPException(500, f"Crop failed: {e}")


@app.get("/dl/{run_id}/file")
async def download_file_public(run_id: str, path: str):
    import re as _re
    if not _re.match(r'^[0-9a-f-]{36}$', run_id):
        raise HTTPException(400, "Invalid run id")
    run_root = (RUNS_DIR / run_id).resolve()
    target = (run_root / path).resolve()
    if not target.is_relative_to(run_root):
        raise HTTPException(400, "Invalid path")
    if not target.exists() or not target.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(target), filename=target.name)


@app.get("/dl/{run_id}/zip")
async def download_zip_public(run_id: str, which: str = "output"):
    import re as _re
    if not _re.match(r'^[0-9a-f-]{36}$', run_id):
        raise HTTPException(400, "Invalid run id")
    run_root = RUNS_DIR / run_id
    if which == "output":
        roots = [_output_dir(run_id)]
    elif which == "uploads":
        roots = [_uploads_dir(run_id)]
    else:
        roots = [_uploads_dir(run_id), _output_dir(run_id)]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root in roots:
            if not root.exists():
                continue
            for f in root.rglob("*"):
                if f.is_file():
                    zf.write(f, f.relative_to(run_root).as_posix())
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{run_id}_{which}.zip"'},
    )


@app.get("/api/runs/{run_id}/viewer/{filepath:path}")
async def serve_viewer(run_id: str, filepath: str, user: dict = Depends(auth.get_current_user)):
    await auth.require_run_owner(run_id, user)
    run_root = _output_dir(run_id).resolve()
    target = (_output_dir(run_id) / "04_viewer" / filepath).resolve()
    if not target.is_relative_to(run_root):
        raise HTTPException(400, "Invalid path")
    if not target.exists():
        raise HTTPException(404, "Viewer not yet generated.")
    return FileResponse(str(target))


@app.get("/viewer/{run_id}/{filepath:path}")
async def serve_viewer_public(run_id: str, filepath: str):
    """
    Public viewer route — no auth header required so it works inside an iframe.
    UUID run_id is unguessable, providing sufficient security for a research tool.
    """
    import re as _re
    if not _re.match(r'^[0-9a-f-]{36}$', run_id):
        raise HTTPException(400, "Invalid run id")
    target = (_output_dir(run_id) / "04_viewer" / filepath).resolve()
    run_root = _output_dir(run_id).resolve()
    if not target.is_relative_to(run_root):
        raise HTTPException(400, "Invalid path")
    if not target.exists():
        raise HTTPException(404, "Viewer file not found.")
    return FileResponse(str(target))


@app.get("/files/{run_id}")
def file_browser_page(run_id: str):
    """Standalone file browser — auth handled client-side via JWT in localStorage."""
    p = _find_page("file-browser.html")
    if not p:
        raise HTTPException(404, "file-browser.html not found")
    return FileResponse(str(p))


# ---------------------------------------------------------------------------
# Pipeline runner (writes status straight to Postgres; logs errors richly)
# ---------------------------------------------------------------------------
def _run_pipeline(run_id: str, owner_email: str, params: dict):
    """
    Runs in a background thread with its OWN event loop and its OWN asyncpg
    connection — never sharing the main uvicorn loop's pool, which would cause
    'Future attached to a different loop' errors.
    """
    import asyncio
    import asyncpg as _asyncpg

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_run_pipeline_async(run_id, owner_email, params, loop))
    finally:
        loop.close()


async def _run_pipeline_async(run_id: str, owner_email: str, params: dict, loop):
    """
    Uses a dedicated asyncpg connection (not the shared pool) so it is safe
    to run inside a background thread with its own event loop.
    """
    import asyncpg as _asyncpg

    # Open a dedicated connection for this pipeline run
    con = await _asyncpg.connect(
        host=db.DB_HOST, port=db.DB_PORT, database=db.DB_NAME,
        user=db.DB_USER, password=db.DB_PASS,
    )

    async def _update(**fields):
        """Update run columns using our dedicated connection."""
        import json as _json
        if not fields:
            return
        if "params" in fields and isinstance(fields["params"], dict):
            fields["params"] = _json.dumps(fields["params"])
        cols = list(fields.keys())
        set_clause = ", ".join(f"{c} = ${i+2}" for i, c in enumerate(cols))
        values = [fields[c] for c in cols]
        await con.execute(
            f"UPDATE runs SET {set_clause} WHERE id = $1",
            run_id, *values,
        )

    async def _get_step():
        row = await con.fetchrow("SELECT current_step FROM runs WHERE id = $1", run_id)
        return row["current_step"] if row else None

    run_log = _log_path(run_id)
    om_dir = str(_uploads_dir(run_id))
    out_dir = str(_output_dir(run_id))

    # Read run_name from DB
    row = await con.fetchrow("SELECT run_name FROM runs WHERE id = $1", run_id)
    run_name = row["run_name"] if row else run_id

    await _update(status="running", current_step="00_discover_oms",
                  started_at=datetime.now(timezone.utc), log_path=str(run_log))

    # Create a job record for this compute attempt
    import uuid as _uuid
    job_id = str(_uuid.uuid4())
    await con.execute("""
        INSERT INTO jobs (id, run_id, type, state, current_stage, started_at)
        VALUES ($1, $2, 'pipeline', 'RUNNING', '00_discover_oms', now());
    """, job_id, run_id)

    async def _update_job(**fields):
        if not fields:
            return
        cols = list(fields.keys())
        set_clause = ", ".join(f"{c} = ${i+2}" for i, c in enumerate(cols))
        values = [fields[c] for c in cols]
        await con.execute(f"UPDATE jobs SET {set_clause} WHERE id = $1", job_id, *values)

    raw_output_parts: list[str] = []
    step_idx = 0
    try:
        cmd = _build_cmd(run_name, om_dir, out_dir, params)
        log.info("Job start  run=%s  job=%s  user=%s", run_id, job_id, owner_email)
        log.debug("Pipeline cmd: %s", " ".join(str(a) for a in cmd))
        log.debug("Pipeline cwd: %s", PIPELINE_SCRIPT.parent)

        with open(run_log, "w", buffering=1, encoding="utf-8") as lf:
            lf.write(f"[server] cmd  : {' '.join(str(a) for a in cmd)}\n")
            lf.write(f"[server] cwd  : {PIPELINE_SCRIPT.parent}\n\n")

            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, cwd=str(PIPELINE_SCRIPT.parent),
            )
            for line in proc.stdout:
                lf.write(line)
                lf.flush()
                raw_output_parts.append(line)
                stripped = line.strip()
                if stripped.startswith("STEP:"):
                    key = stripped[5:].strip()
                    if key in STEP_KEYS:
                        step_idx = STEP_KEYS.index(key)
                        progress = round(_STEP_START.get(key, step_idx / len(STEP_KEYS)), 4)
                        await _update(current_step=key, step_progress=progress)
                        await _update_job(current_stage=key, progress=progress)
                        _write_progress(run_id, key, progress, 0, 0,
                                        STEP_LABELS.get(key, key))
                        log.debug("Step  run=%s  step=%s  progress=%.0f%%", run_id, key, progress * 100)
                        lf.write(f"[server] >> current_step = {key} ({progress:.0%})\n")
                        lf.flush()
                elif stripped.startswith("PROGRESS:"):
                    # PROGRESS:<step_key>:<cur>/<tot>:<label>
                    body = stripped[len("PROGRESS:"):]
                    parts = body.split(":", 2)
                    if len(parts) >= 2 and parts[0] in STEP_WEIGHTS:
                        p_step = parts[0]
                        frac = parts[1].strip()
                        label = parts[2].strip() if len(parts) >= 3 else STEP_LABELS.get(p_step, p_step)
                        cur = tot = 0
                        within = 0.0
                        if "/" in frac:
                            cs, _, ts = frac.partition("/")
                            try:
                                cur = int(cs); tot = int(ts)
                                within = (cur / tot) if tot > 0 else 0.0
                            except ValueError:
                                within = 0.0
                        within = max(0.0, min(1.0, within))
                        step_start = _STEP_START.get(p_step, 0.0)
                        progress = round(step_start + within * STEP_WEIGHTS.get(p_step, 0.0), 4)
                        await _update(current_step=p_step, step_progress=progress)
                        _write_progress(run_id, p_step, progress, cur, tot, label)
            proc.wait()

        if proc.returncode == 0:
            log.info("Job complete  run=%s  job=%s  state=SUCCEEDED", run_id, job_id)
            await _update(status="done", step_progress=1.0,
                          finished_at=datetime.now(timezone.utc))
            await _update_job(state="SUCCEEDED", progress=1.0,
                              finished_at=datetime.now(timezone.utc))
            _write_progress(run_id, "04b_interactive_viz", 1.0, 0, 0, "Complete")
            _generate_thumbnail(run_id)   # best-effort; logged on failure, never raises
        else:
            raw = "".join(raw_output_parts)
            current_step = await _get_step()
            info = errors.log_error(
                run_id=run_id, user_email=owner_email,
                step=current_step, raw_output=raw, message=None,
            )
            log.error(
                "Job failed  run=%s  job=%s  step=%s  summary=%s",
                run_id, job_id, current_step, info["summary"],
            )
            await _update(
                status="failed",
                error_msg=info["summary"],
                finished_at=datetime.now(timezone.utc),
            )
            await _update_job(
                state="FAILED",
                error=info["summary"],
                finished_at=datetime.now(timezone.utc),
            )

    except Exception as e:                           # noqa: BLE001
        raw = "".join(raw_output_parts)
        info = errors.log_error(
            run_id=run_id, user_email=owner_email,
            step=None, exc=e, raw_output=raw or None,
        )
        log.error(
            "Job exception  run=%s  job=%s  error=%s",
            run_id, job_id, info["summary"], exc_info=True,
        )
        try:
            with open(run_log, "a", encoding="utf-8") as lf:
                lf.write(f"\n[ERROR] {info['summary']}\n")
        except Exception:                            # noqa: BLE001
            pass
        await _update(
            status="failed",
            error_msg=info["summary"],
            finished_at=datetime.now(timezone.utc),
        )
        try:
            await _update_job(
                state="FAILED",
                error=info["summary"],
                finished_at=datetime.now(timezone.utc),
            )
        except Exception:
            pass
    finally:
        await con.close()


def _build_cmd(run_name: str, om_dir: str, out_dir: str, params: dict) -> list:
    if _ON_WINDOWS:
        return _build_windows_cmd(run_name, om_dir, out_dir, params)
    return _build_bash_cmd(run_name, om_dir, out_dir, params)


def _build_bash_cmd(run_name: str, om_dir: str, out_dir: str, params: dict) -> list:
    args = ["bash", str(PIPELINE_SCRIPT),
            "--om-dir", om_dir, "--output-dir", out_dir, "--run-name", run_name]
    if PROJECT_ROOT:
        args += ["--project-root", str(PROJECT_ROOT)]
    model_path = _resolve_model_path(params)
    if model_path:
        args += ["--model-path", model_path]

    def _sh(key, flag):
        v = params.get(key, "")
        if v:
            args.extend([flag, str(v)])

    def _sh_int(key, flag):
        v = params.get(key)
        if v is not None:
            args.extend([flag, str(int(v))])

    def _sh_unit_float(key, flag):
        """Pass a 0..1 number through; ignore anything else (e.g. an unset
        Airflow param arriving as a placeholder string) so the pipeline
        falls back to its own default instead of crashing."""
        v = params.get(key)
        if v is None or v == "":
            return
        try:
            f = float(v)
        except (TypeError, ValueError):
            log.warning("Ignoring non-numeric %s=%r", key, v)
            return
        if not 0.0 <= f <= 1.0:
            log.warning("Ignoring out-of-range %s=%r (expected 0..1)", key, v)
            return
        args.extend([flag, repr(f)])

    _sh_int("tile_width", "--tile-width")
    _sh_int("tile_height", "--tile-height")
    _sh_int("tile_buffer", "--tile-buffer")
    _sh_unit_float("fixed_iou", "--fixed-iou")

    def _sh_float(key, flag):
        """Pass any finite number through — including 0 and negatives, which
        the generic _sh() helper below would drop or mangle. Non-numbers are
        ignored so the pipeline default applies."""
        v = params.get(key)
        if v is None or v == "" or isinstance(v, bool):
            return
        try:
            f = float(v)
        except (TypeError, ValueError):
            log.warning("Ignoring non-numeric %s=%r", key, v)
            return
        if f != f or f in (float("inf"), float("-inf")):
            log.warning("Ignoring non-finite %s=%r", key, v)
            return
        args.extend([flag, repr(f)])

    # Deciduousness-score weights (and optional threshold) -> steps 3 and 3b
    _sh_float("w_veg_amp", "--w-veg-amp")
    _sh_float("w_depth", "--w-depth")
    _sh_float("w_gcc_amp", "--w-gcc-amp")
    _sh_float("w_tex", "--w-tex")
    _sh_float("ds_threshold", "--ds-thresh")
    _sh("align_method", "--align-method")
    _sh("base_threshold_tag", "--base-threshold-tag")
    _sh("align_threshold_tag", "--align-threshold-tag")
    _sh("min_partial_len", "--min-partial-len")
    _sh("min_partial_ratio", "--min-partial-ratio")
    _sh("exclude_stems", "--exclude-stems")
    _sh("crowns_dir", "--crowns-dir")
    _sh("steps", "--steps")
    _sh("base_env", "--base-env")
    _sh("tracking_env", "--tracking-env")
    if params.get("skip_existing") is False:
        args.append("--no-skip-existing")
    if params.get("skip_chain_viz"):
        args.append("--skip-chain-viz")
    if params.get("skip_consensus_viz"):
        args.append("--skip-consensus-viz")
    return args


def _build_windows_cmd(run_name: str, om_dir: str, out_dir: str, params: dict) -> list:
    ps = [f"-OmDir '{om_dir}'", f"-OutputDir '{out_dir}'", f"-RunName '{run_name}'"]
    if PROJECT_ROOT:
        ps.append(f"-ProjectRoot '{PROJECT_ROOT}'")
    model_path = _resolve_model_path(params)
    if model_path:
        ps.append(f"-ModelPath '{model_path}'")

    def _ps_str(key, flag):
        v = params.get(key, "")
        if v:
            ps.append(f"-{flag} '{v}'")

    def _ps_int(key, flag):
        v = params.get(key)
        if v is not None:
            ps.append(f"-{flag} {int(v)}")

    _ps_int("tile_width", "TileWidth")
    _ps_int("tile_height", "TileHeight")
    _ps_int("tile_buffer", "TileBuffer")
    _ps_str("align_method", "AlignMethod")
    _ps_str("base_threshold_tag", "BaseThresholdTag")
    _ps_str("align_threshold_tag", "AlignThresholdTag")
    _ps_str("min_partial_len", "MinPartialLen")
    _ps_str("min_partial_ratio", "MinPartialRatio")
    _ps_str("exclude_stems", "ExcludeStems")
    _ps_str("crowns_dir", "CrownsDir")
    _ps_str("steps", "Steps")
    _ps_str("base_env", "BaseEnv")
    _ps_str("tracking_env", "TrackingEnv")
    if params.get("skip_existing") is False:
        ps.append("-SkipExisting '--no-skip-existing'")
    if params.get("skip_chain_viz"):
        ps.append("-SkipChainViz")
    if params.get("skip_consensus_viz"):
        ps.append("-SkipConsensusViz")

    flat = []
    for part in ps:
        if " " in part:
            flag, _, rest = part.partition(" ")
            flat.extend([flag, rest.strip().strip("'").strip('"')])
        else:
            flat.append(part)
    return ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(PIPELINE_SCRIPT)] + flat


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
