"""
Structured, human-readable error logging.

Two sinks:
  - a central JSON-lines error log  (errors.jsonl) with full tracebacks
  - per-run pipeline logs already written by the pipeline itself

The key idea: never surface a bare "exit code 1" or a truncated message. Every
error carries the run id, the step it failed at, the exception type, the clean
message, and the full traceback. classify_pipeline_error() turns common raw
stderr into a plain-English explanation.
"""
import json
import re
import traceback
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
ERRORS_JSONL = LOG_DIR / "errors.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_error(
    *,
    run_id: str | None,
    user_email: str | None,
    step: str | None,
    exc: BaseException | None = None,
    message: str | None = None,
    raw_output: str | None = None,
) -> dict:
    """
    Record one error. Returns a dict suitable for returning to the frontend
    (it contains a clean 'summary' plus the detail needed to debug).
    """
    exc_type = type(exc).__name__ if exc else None
    tb = "".join(traceback.format_exception(exc)) if exc else None

    # Build a human summary: prefer an explicit message, else classify raw output,
    # else use the exception's own message.
    if message:
        summary = message
    elif raw_output:
        summary = classify_pipeline_error(raw_output)
    elif exc:
        summary = f"{exc_type}: {exc}"
    else:
        summary = "Unknown error"

    record = {
        "ts": _now_iso(),
        "run_id": run_id,
        "user_email": user_email,
        "step": step,
        "exc_type": exc_type,
        "summary": summary,
        "traceback": tb,
        # keep only the tail of raw output so the file doesn't explode
        "raw_tail": (raw_output[-4000:] if raw_output else None),
    }

    try:
        with open(ERRORS_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:                           # noqa: BLE001
        print(f"[errors] failed to write error log: {e}")

    # Console copy so `docker compose logs` shows it too
    print(f"[ERROR] run={run_id} step={step} :: {summary}")
    if tb:
        print(tb)

    return {
        "summary": summary,
        "step": step,
        "exc_type": exc_type,
        "ts": record["ts"],
    }


# ---------------------------------------------------------------------------
# Turn raw pipeline stderr into plain-English explanations.
# Each entry: (regex, explanation). First match wins.
# ---------------------------------------------------------------------------
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Found no NVIDIA driver", re.I),
     "No GPU available to the container. Start it with `--gpus all`, or set the "
     "pipeline device to CPU."),
    (re.compile(r"CUDA out of memory", re.I),
     "The GPU ran out of memory. Try a smaller tile size, or free other GPU processes."),
    (re.compile(r"No module named ['\"]?mercantile", re.I),
     "The 'mercantile' package is missing in the tracking environment. "
     "Install it: pip install mercantile."),
    (re.compile(r"No module named ['\"]?detectron2", re.I),
     "detectron2 is not installed in the detection environment."),
    (re.compile(r"No module named ['\"]?torch", re.I),
     "PyTorch (torch) is not installed in this environment."),
    (re.compile(r"Checkpoint .* not found", re.I),
     "The detector model weights (.pth) could not be found. Check the model path "
     "in the run configuration / .env."),
    (re.compile(r"EnvironmentLocationNotFound.*?envs[/\\]([A-Za-z0-9_\-]+)", re.I),
     "A required conda environment is missing: \\1. Create it before running."),
    (re.compile(r"No such file or directory: ['\"]?(.+?\.json)", re.I),
     "A file the pipeline expected is missing: \\1. A previous step may have "
     "failed to produce it, or a path exceeds the OS limit."),
    (re.compile(r"unrecognized arguments: (.+)", re.I),
     "The pipeline passed an argument a step doesn't accept: \\1. "
     "This is a wiring bug between the runner and that step."),
    (re.compile(r"Permission denied", re.I),
     "Permission denied accessing a file or directory. Check the mounted volume's "
     "ownership/permissions."),
    (re.compile(r"conda run in '([^']+)' exited with code (\d+)", re.I),
     "The step running in conda env '\\1' failed (exit code \\2). See the traceback "
     "just above this line for the underlying Python error."),
]


def classify_pipeline_error(raw: str) -> str:
    """
    Scan raw stderr/stdout for a known failure signature and return a clear
    explanation. Falls back to the last non-empty, non-warning line.
    """
    for pattern, explanation in _PATTERNS:
        m = pattern.search(raw)
        if m:
            # Substitute any capture groups referenced in the explanation
            try:
                return pattern.sub(explanation, m.group(0))
            except re.error:
                return explanation

    # Fallback: last meaningful line (skip tqdm bars, warnings, blank lines)
    for line in reversed(raw.strip().splitlines()):
        s = line.strip()
        if not s:
            continue
        if s.startswith(("WARNING", "warnings.warn", "INFO:")):
            continue
        if re.match(r"^\d+%\|", s):                  # tqdm progress bar
            continue
        return s

    return "Pipeline failed with no captured error message."


def extract_python_traceback(raw: str) -> str | None:
    """Pull the last Python 'Traceback (most recent call last):' block from raw output."""
    idx = raw.rfind("Traceback (most recent call last):")
    if idx == -1:
        return None
    return raw[idx:].strip()
