"""
airflow_client.py — thin REST client for triggering and polling the Drone
Phenology Monitoring Airflow DAG (cluster checklist §7/§8).

Compute-mode switch (see .env): AIRFLOW_API_BASE is THE switch.
  empty  -> pipeline runs locally, inside this container
            (server.py's existing /api/runs/{run_id}/start background task)
  set    -> the dashboard triggers/polls this DAG via the Airflow REST API
            (trigger_conf / run_state below), and the DAG's single task
            calls this container back over HTTP at
            CORESTACK_API_BASE + /api/export-phenology to do the actual
            compute (see the /api/export-phenology endpoint added to
            server.py).

Auth: set AIRFLOW_USERNAME + AIRFLOW_PASSWORD (HTTP Basic — Airflow's
stable REST API default) OR AIRFLOW_TOKEN (Bearer). Set one, not both.

Requires `httpx` (add to requirements: httpx>=0.27).
"""
from __future__ import annotations

import asyncio
import base64
import os
from datetime import datetime, timezone
from typing import Optional

import httpx

AIRFLOW_API_BASE = os.environ.get("AIRFLOW_API_BASE", "").rstrip("/")
AIRFLOW_DAG_ID = os.environ.get("AIRFLOW_DAG_ID", "drone_phenology_monitoring")
AIRFLOW_USERNAME = os.environ.get("AIRFLOW_USERNAME", "")
AIRFLOW_PASSWORD = os.environ.get("AIRFLOW_PASSWORD", "")
AIRFLOW_TOKEN = os.environ.get("AIRFLOW_TOKEN", "")

# THE switch other modules should check before using this client at all.
AIRFLOW_ENABLED = bool(AIRFLOW_API_BASE)

# Airflow REST API dag_run states -> DPM's normalized job states
# (matches the `jobs.state` values server.py already writes for local runs:
#  QUEUED / RUNNING / SUCCEEDED / FAILED)
_STATE_MAP = {
    "queued":  "QUEUED",
    "running": "RUNNING",
    "success": "SUCCEEDED",
    "failed":  "FAILED",
}


class AirflowClientError(RuntimeError):
    """Raised for any non-2xx response from the Airflow REST API, or when
    this client is called while AIRFLOW_API_BASE is unset."""


def _auth_headers() -> dict:
    if AIRFLOW_TOKEN:
        return {"Authorization": f"Bearer {AIRFLOW_TOKEN}"}
    if AIRFLOW_USERNAME:
        basic = base64.b64encode(
            f"{AIRFLOW_USERNAME}:{AIRFLOW_PASSWORD}".encode()
        ).decode()
        return {"Authorization": f"Basic {basic}"}
    return {}


def _client() -> httpx.AsyncClient:
    if not AIRFLOW_ENABLED:
        raise AirflowClientError(
            "AIRFLOW_API_BASE is not set — this deployment runs the "
            "pipeline locally, not via Airflow."
        )
    return httpx.AsyncClient(
        base_url=AIRFLOW_API_BASE,
        headers={"Content-Type": "application/json", **_auth_headers()},
        timeout=30.0,
    )


async def trigger_conf(conf: dict, dag_run_id: Optional[str] = None) -> str:
    """
    POST to Airflow's stable REST API to start a new run of AIRFLOW_DAG_ID
    with the given `conf` payload (the params dpm_dag.yaml declares: run_id,
    state, district, year, block, bbox).

    Returns Airflow's dag_run_id — store this (server.py stores it on the
    job row's existing `request_id` column) so run_state() can poll it later.
    """
    if dag_run_id is None:
        dag_run_id = f"dpm_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')}"
    body = {"dag_run_id": dag_run_id, "conf": conf}
    async with _client() as client:
        resp = await client.post(f"/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns", json=body)
    if resp.status_code not in (200, 201):
        raise AirflowClientError(
            f"Airflow trigger failed ({resp.status_code}): {resp.text[:500]}"
        )
    data = resp.json()
    return data.get("dag_run_id", dag_run_id)


async def run_state(dag_run_id: str) -> dict:
    """
    GET the current state of a dag_run.

    Returns:
        {"state": "QUEUED" | "RUNNING" | "SUCCEEDED" | "FAILED" | "UNKNOWN",
         "airflow_state": <raw Airflow state string, or None>,
         "start_date": <str|None>, "end_date": <str|None>}
    """
    async with _client() as client:
        resp = await client.get(f"/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns/{dag_run_id}")
    if resp.status_code == 404:
        return {"state": "UNKNOWN", "airflow_state": None,
                "start_date": None, "end_date": None}
    if resp.status_code != 200:
        raise AirflowClientError(
            f"Airflow status check failed ({resp.status_code}): {resp.text[:500]}"
        )
    data = resp.json()
    raw_state = data.get("state", "")
    return {
        "state": _STATE_MAP.get(raw_state, "UNKNOWN"),
        "airflow_state": raw_state,
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
    }


async def poll_until_done(dag_run_id: str, interval: float = 3.0,
                          timeout: float = 3600.0) -> dict:
    """
    Convenience helper: poll run_state() every `interval`s until it leaves
    QUEUED/RUNNING or `timeout` is hit. NOT used by the HTTP endpoints
    (those poll once per UI request instead, via GET /api/dag/status) —
    handy for a CLI integration test or a script.
    """
    elapsed = 0.0
    while elapsed < timeout:
        st = await run_state(dag_run_id)
        if st["state"] in ("SUCCEEDED", "FAILED", "UNKNOWN"):
            return st
        await asyncio.sleep(interval)
        elapsed += interval
    raise AirflowClientError(f"Timed out after {timeout}s waiting on {dag_run_id}")
