"""FileBrowser REST client for Drone Phenology Monitoring.

Creates permanent public shares for run folders so users can browse and
download pipeline outputs from one link without logging in.

Adapted from TreeCrownDetection's filebrowser_client.py.

With DPM_FILEBROWSER_BASE_URL unset (default), this module does nothing —
all functions degrade gracefully to no-ops / None returns.
"""
import json
import urllib.error
import urllib.request
import urllib.parse
import os

from log_config import get_logger

log = get_logger("filebrowser")

# ── Settings from environment ──────────────────────────────────────────────
_BASE_URL = (os.environ.get("DPM_FILEBROWSER_BASE_URL") or "").strip().rstrip("/")
_PUBLIC_URL = (os.environ.get("DPM_FILEBROWSER_PUBLIC_URL") or "").strip().rstrip("/")
_USERNAME = os.environ.get("DPM_FILEBROWSER_USERNAME") or "admin"
_PASSWORD = os.environ.get("DPM_FILEBROWSER_PASSWORD") or ""
_RUNS_PREFIX = (os.environ.get("DPM_FILEBROWSER_RUNS_PATH") or "").strip().strip("/")


def filebrowser_enabled() -> bool:
    """True when a FileBrowser base URL is configured."""
    return bool(_BASE_URL)


def _get_token() -> str:
    """Log in to FileBrowser and return the JWT token."""
    body = json.dumps({
        "username": _USERNAME,
        "password": _PASSWORD,
    }).encode()
    req = urllib.request.Request(
        f"{_BASE_URL}/api/login",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.read().decode().strip()


def create_run_share(run_id: str) -> str | None:
    """Create a permanent public share for one run's folder.

    Returns the share hash (e.g. 'fNqIKDS3'), or None on failure.
    The run_id is used as the folder name under /srv in FileBrowser,
    matching RUNS_DIR layout: each run is a UUID folder.
    """
    if not filebrowser_enabled():
        return None

    try:
        token = _get_token()
    except urllib.error.HTTPError as e:
        log.error("FileBrowser login failed: HTTP %s", e.code, exc_info=True)
        return None
    except urllib.error.URLError as e:
        log.error("FileBrowser unreachable at %s: %s", _BASE_URL, e.reason, exc_info=True)
        return None
    except Exception as e:
        log.error("FileBrowser login failed: %s", e, exc_info=True)
        return None

    # Build the share path relative to /srv
    share_path = f"{_RUNS_PREFIX}/{run_id}" if _RUNS_PREFIX else run_id
    body = json.dumps({}).encode()
    req = urllib.request.Request(
        f"{_BASE_URL}/api/share/{urllib.parse.quote(share_path)}",
        data=body,
        headers={"Content-Type": "application/json", "X-Auth": token},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        share_hash = data["hash"]
        log.info("Created FileBrowser share for run %s: %s", run_id, share_hash)
        return share_hash
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        log.error("FileBrowser share API error %s: %s", e.code, detail, exc_info=True)
        return None
    except urllib.error.URLError as e:
        log.error("FileBrowser unreachable at %s: %s", _BASE_URL, e.reason, exc_info=True)
        return None
    except Exception as e:
        log.error("FileBrowser share failed: %s", e, exc_info=True)
        return None


def share_url(share_hash: str) -> str:
    """Build the user-facing share URL from a stored hash.

    Uses DPM_FILEBROWSER_PUBLIC_URL (the address the browser can reach)
    when set, falling back to DPM_FILEBROWSER_BASE_URL.
    """
    base = (_PUBLIC_URL or _BASE_URL).rstrip("/")
    return f"{base}/share/{share_hash}"
