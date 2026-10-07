"""
Application logging — CoRE Stack cluster checklist §5.

Writes to  data/logs/drone-phenology-monitoring/  (bind-mounted from the host)
AND to stdout (so `docker compose logs -f dashboard` also works).

LOG_LEVEL env var controls granularity:
    debug  — request traces, job params, Airflow poll, filesystem paths
    info   — (default) start-up, auth, compute trigger, job start/complete
    error  — failures only: exceptions, failed jobs, SSO/config errors

Security: no tokens, passwords, or Google client secrets at ANY level.
"""

import logging
import os
import sys
import time
from datetime import datetime, timezone, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APP_NAME = "drone-phenology-monitoring"

# DPM_LOG_DIR is set in docker-compose.yml → /app/data/logs/drone-phenology-monitoring
# Fallback: data/logs/<APP_NAME> relative to the code dir (works for local dev too).
_DEFAULT_LOG_DIR = Path(__file__).parent.resolve() / "data" / "logs" / APP_NAME
LOG_DIR = Path(os.environ.get("DPM_LOG_DIR", str(_DEFAULT_LOG_DIR)))

# Map the three CoRE Stack levels to Python logging levels.
_LEVEL_MAP = {
    "debug": logging.DEBUG,
    "info":  logging.INFO,
    "error": logging.ERROR,
}


def _resolve_level() -> int:
    raw = os.environ.get("LOG_LEVEL", "info").strip().lower()
    return _LEVEL_MAP.get(raw, logging.INFO)


# ---------------------------------------------------------------------------
# Timezone — LOG_TIMEZONE env var (e.g. "Asia/Kolkata", "+05:30", "UTC").
# Defaults to UTC so cluster logs are consistent.
# ---------------------------------------------------------------------------
def _resolve_tz_offset() -> timezone:
    """Parse LOG_TIMEZONE into a datetime.timezone for the formatter."""
    raw = os.environ.get("LOG_TIMEZONE", "UTC").strip()
    if raw.upper() == "UTC":
        return timezone.utc
    # Try common named offsets: "Asia/Kolkata" -> +05:30, "IST" -> +05:30
    _NAMED = {
        "Asia/Kolkata": timedelta(hours=5, minutes=30),
        "IST":          timedelta(hours=5, minutes=30),
        "Asia/Calcutta": timedelta(hours=5, minutes=30),
    }
    if raw in _NAMED:
        return timezone(_NAMED[raw])
    # Try explicit offset: "+05:30", "-04:00"
    try:
        sign = 1 if raw[0] != "-" else -1
        parts = raw.lstrip("+-").split(":")
        hours = int(parts[0])
        minutes = int(parts[1]) if len(parts) > 1 else 0
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    except (ValueError, IndexError):
        return timezone.utc


_LOG_TZ = _resolve_tz_offset()


class _TZFormatter(logging.Formatter):
    """Formatter that stamps every record in the configured LOG_TIMEZONE."""
    converter = time.gmtime  # base on UTC, then shift

    def formatTime(self, record, datefmt=None):  # noqa: N802
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone(_LOG_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()


# ---------------------------------------------------------------------------
# Format string
# ---------------------------------------------------------------------------
_FMT = "%(asctime)s  %(levelname)-5s  %(name)s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


# ---------------------------------------------------------------------------
# Setup (call once at import / startup)
# ---------------------------------------------------------------------------
def setup_logging() -> logging.Logger:
    """
    Configure the root 'dpm' logger with:
      • a RotatingFileHandler  → data/logs/drone-phenology-monitoring/app.log
      • a StreamHandler         → stdout (captured by `docker compose logs`)

    Returns the 'dpm' logger.  Sub-modules use  logging.getLogger("dpm.auth")  etc.
    """
    level = _resolve_level()

    # Ensure the log directory exists (also done by docker-compose command, but
    # belt-and-suspenders for local dev without Docker).
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Root application logger
    logger = logging.getLogger("dpm")
    logger.setLevel(level)

    # Avoid duplicate handlers on hot-reload
    if logger.handlers:
        return logger

    formatter = _TZFormatter(_FMT, datefmt=_DATEFMT)

    # --- File handler (rotates at 10 MB, keeps 5 backups) ---
    log_file = LOG_DIR / "app.log"
    fh = RotatingFileHandler(
        str(log_file),
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # --- Stdout handler (for `docker compose logs -f`) ---
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # Quiet down noisy third-party loggers unless we're in debug
    for noisy in ("uvicorn", "uvicorn.access", "uvicorn.error",
                  "asyncpg", "httpcore", "httpx"):
        third = logging.getLogger(noisy)
        third.setLevel(logging.DEBUG if level == logging.DEBUG else logging.WARNING)

    logger.info(
        "Logging initialised  level=%s  tz=%s  dir=%s  file=%s",
        logging.getLevelName(level), _LOG_TZ, LOG_DIR, log_file,
    )
    return logger


# ---------------------------------------------------------------------------
# Convenience: importable pre-configured logger
# ---------------------------------------------------------------------------
logger = setup_logging()


def get_logger(name: str) -> logging.Logger:
    """Return a child logger, e.g.  get_logger('auth') → 'dpm.auth'."""
    return logging.getLogger(f"dpm.{name}")
