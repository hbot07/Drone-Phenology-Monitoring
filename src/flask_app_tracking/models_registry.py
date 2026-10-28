"""Model catalog, loaded from an external YAML manifest (models.yaml).

The manifest holds one section:

* detectors: Detectree2/detectron2 ``.pth`` weight files used for crown
  detection, selected per run through ``model_type``. Each ``file`` is
  looked up under the configured models directory (DPM_MODELS_DIR, or the
  folder that holds DPM_MODEL_PATH, or <project>/models).

The catalog path comes from the ``DPM_MODELS_MANIFEST`` env var (default:
``models.yaml`` next to server.py), so it can be mounted as a Docker volume
and edited without rebuilding the image.

If the file is missing, ``_FALLBACK`` below is used and the app still starts.
"""
import os
from functools import lru_cache
from pathlib import Path

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False

# ---------------------------------------------------------------------------
# Fallback catalog — used when models.yaml is missing or PyYAML not installed.
# Should match the default entry in models.yaml.
# ---------------------------------------------------------------------------
_FALLBACK: dict = {
    "detectors": {
        "250312_flexi": {
            "file": "250312_flexi.pth",
            "description": "Flexible multi-site model (recommended default)",
            "default": True,
        },
    },
}


def _models_dir() -> Path:
    """Return the directory where .pth files live (same priority as _resolve_model_path)."""
    if os.environ.get("DPM_MODELS_DIR"):
        return Path(os.environ["DPM_MODELS_DIR"])
    model_path = os.environ.get("DPM_MODEL_PATH", "")
    if model_path:
        return Path(model_path).parent
    return Path(__file__).parent / "models"


@lru_cache(maxsize=1)
def _manifest() -> dict:
    manifest_path = os.environ.get(
        "DPM_MODELS_MANIFEST",
        str(Path(__file__).parent / "models.yaml"),
    )
    if manifest_path and os.path.exists(manifest_path) and _YAML_OK:
        with open(manifest_path) as f:
            data = _yaml.safe_load(f) or {}
        data.setdefault("detectors", {})
        return data
    return _FALLBACK


def _detectors() -> dict:
    return _manifest()["detectors"]


def _default_key() -> str:
    for key, meta in _detectors().items():
        if meta.get("default"):
            return key
    # Fallback: env var stem, then first entry
    env_stem = Path(os.environ.get("DPM_MODEL_PATH", "")).stem
    if env_stem and env_stem in _detectors():
        return env_stem
    return next(iter(_detectors()))


def list_models() -> list[dict]:
    """Return detectors with availability and default flags, for the UI."""
    mdir = _models_dir()
    default_key = _default_key()
    out = []
    for key, meta in _detectors().items():
        fname = meta["file"]
        available = (mdir / fname).exists()
        out.append({
            "key": key,
            "filename": fname,
            "description": meta.get("description", ""),
            "available": available,
            "is_default": key == default_key,
        })
    return out


def resolve_model_path(model_key: str | None) -> str:
    """Return the absolute path for a model_key, or the default. Empty string if not found."""
    detectors = _detectors()
    key = model_key if model_key and model_key in detectors else _default_key()
    fname = detectors[key]["file"]
    path = ((_models_dir()) / fname).resolve()
    return str(path) if path.exists() else ""
