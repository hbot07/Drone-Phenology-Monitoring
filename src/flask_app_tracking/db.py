"""
Database layer — raw asyncpg queries against Postgres.

Tables:
  users   : one row per account (Google or email/password)
  runs    : one row per pipeline run, owned by a user
  orthos  : one row per uploaded orthomosaic, owned by a run
  jobs    : one row per compute attempt (pipeline execution), owned by a run

Schema is created on startup by init_db(). No ORM — plain SQL.
"""
import os
import json
import uuid
from datetime import datetime, timezone, date

import asyncpg

# ---------------------------------------------------------------------------
# Connection settings
# ---------------------------------------------------------------------------
DB_HOST = os.environ.get("DPM_DB_HOST", "dpm-postgres")
DB_PORT = int(os.environ.get("DPM_DB_PORT", "5432"))
DB_NAME = os.environ["DPM_DB_NAME"]
DB_USER = os.environ["DPM_DB_USER"]
DB_PASS = os.environ["DPM_DB_PASSWORD"]

_pool: asyncpg.Pool | None = None


async def connect(retries: int = 10, delay: float = 2.0):
    global _pool
    import asyncio
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            _pool = await asyncpg.create_pool(
                host=DB_HOST, port=DB_PORT, database=DB_NAME,
                user=DB_USER, password=DB_PASS,
                min_size=1, max_size=10,
            )
            print(f"[db] connected to {DB_HOST}:{DB_PORT}/{DB_NAME}")
            return
        except Exception as e:
            last_err = e
            print(f"[db] connect attempt {attempt}/{retries} failed: {e}")
            await asyncio.sleep(delay)
    raise RuntimeError(f"Could not connect to Postgres after {retries} tries: {last_err}")


async def disconnect():
    if _pool:
        await _pool.close()


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("DB pool not initialised — call connect() first")
    return _pool


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
async def init_db():
    """Create tables if they don't exist. Safe to run on every boot."""
    async with pool().acquire() as con:
        # ---- users ----
        await con.execute("""
            CREATE TABLE IF NOT EXISTS users (
                email         TEXT PRIMARY KEY,
                name          TEXT,
                picture       TEXT,
                password_hash TEXT,
                auth_provider TEXT NOT NULL DEFAULT 'password',
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                last_seen_at  TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)

        # ---- runs ----
        await con.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id            TEXT PRIMARY KEY,
                owner_email   TEXT NOT NULL REFERENCES users(email),
                run_name      TEXT NOT NULL,
                status        TEXT NOT NULL DEFAULT 'created',
                current_step  TEXT,
                step_progress FLOAT NOT NULL DEFAULT 0.0,
                error_msg     TEXT,
                params        JSONB NOT NULL DEFAULT '{}'::jsonb,
                num_orthos    INTEGER NOT NULL DEFAULT 0,
                total_bytes   BIGINT NOT NULL DEFAULT 0,
                log_path      TEXT,
                created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
                started_at    TIMESTAMPTZ,
                finished_at   TIMESTAMPTZ,
                updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        await con.execute("CREATE INDEX IF NOT EXISTS idx_runs_owner ON runs(owner_email);")
        await con.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);")

        # ---- orthos (uploaded orthomosaics) ----
        await con.execute("""
            CREATE TABLE IF NOT EXISTS orthos (
                id                 TEXT PRIMARY KEY,
                run_id             TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                original_filename  TEXT NOT NULL,
                pipeline_filename  TEXT,
                acquisition_date   DATE,
                bands              INTEGER,
                size_bytes         BIGINT,
                upload_order       INTEGER NOT NULL DEFAULT 0,
                created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        await con.execute("CREATE INDEX IF NOT EXISTS idx_orthos_run ON orthos(run_id);")

        # ---- jobs (compute attempts) ----
        await con.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id            TEXT PRIMARY KEY,
                run_id        TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                type          TEXT NOT NULL DEFAULT 'pipeline',
                state         TEXT NOT NULL DEFAULT 'QUEUED',
                current_stage TEXT,
                progress      FLOAT NOT NULL DEFAULT 0.0,
                error         TEXT,
                request_id    TEXT,
                log_path      TEXT,
                started_at    TIMESTAMPTZ,
                finished_at   TIMESTAMPTZ
            );
        """)
        await con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_run ON jobs(run_id);")
        await con.execute("CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);")

        # ---- Migrations for existing databases ----
        for col_sql in [
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT;",
            "ALTER TABLE users ADD COLUMN IF NOT EXISTS auth_provider TEXT NOT NULL DEFAULT 'password';",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS step_progress FLOAT NOT NULL DEFAULT 0.0;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS num_orthos INTEGER NOT NULL DEFAULT 0;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS total_bytes BIGINT NOT NULL DEFAULT 0;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS log_path TEXT;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS started_at TIMESTAMPTZ;",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now();",
        ]:
            try:
                await con.execute(col_sql)
            except Exception:
                pass  # column already exists with different constraint — fine

    print("[db] schema ready (users, runs, orthos, jobs)")


# ═══════════════════════════════════════════════════════════════════════════
# Users
# ═══════════════════════════════════════════════════════════════════════════
async def upsert_user(email: str, name: str | None, picture: str | None,
                      auth_provider: str = "google"):
    async with pool().acquire() as con:
        await con.execute("""
            INSERT INTO users (email, name, picture, auth_provider, last_seen_at)
            VALUES ($1, $2, $3, $4, now())
            ON CONFLICT (email) DO UPDATE
              SET name = COALESCE(EXCLUDED.name, users.name),
                  picture = COALESCE(EXCLUDED.picture, users.picture),
                  last_seen_at = now();
        """, email, name, picture, auth_provider)


async def get_user(email: str) -> dict | None:
    async with pool().acquire() as con:
        row = await con.fetchrow("SELECT * FROM users WHERE email = $1", email)
    return dict(row) if row else None


async def create_password_user(email: str, name: str | None, password_hash: str):
    async with pool().acquire() as con:
        await con.execute("""
            INSERT INTO users (email, name, password_hash, auth_provider, last_seen_at)
            VALUES ($1, $2, $3, 'password', now());
        """, email, name, password_hash)


async def touch_user(email: str):
    async with pool().acquire() as con:
        await con.execute("UPDATE users SET last_seen_at = now() WHERE email = $1", email)


# ═══════════════════════════════════════════════════════════════════════════
# Runs
# ═══════════════════════════════════════════════════════════════════════════
async def create_run(run_id: str, owner_email: str, run_name: str):
    async with pool().acquire() as con:
        await con.execute("""
            INSERT INTO runs (id, owner_email, run_name, status, created_at, updated_at)
            VALUES ($1, $2, $3, 'created', now(), now());
        """, run_id, owner_email, run_name)


async def get_run(run_id: str) -> dict | None:
    async with pool().acquire() as con:
        row = await con.fetchrow("SELECT * FROM runs WHERE id = $1", run_id)
    return _row_to_run(row) if row else None


async def list_runs_for_user(owner_email: str) -> list[dict]:
    async with pool().acquire() as con:
        rows = await con.fetch(
            "SELECT * FROM runs WHERE owner_email = $1 ORDER BY created_at DESC",
            owner_email,
        )
    return [_row_to_run(r) for r in rows]


async def update_run(run_id: str, **fields):
    if not fields:
        return
    if "params" in fields and isinstance(fields["params"], dict):
        fields["params"] = json.dumps(fields["params"])
    fields["updated_at"] = datetime.now(timezone.utc)
    cols = list(fields.keys())
    set_clause = ", ".join(f"{c} = ${i+2}" for i, c in enumerate(cols))
    values = [fields[c] for c in cols]
    async with pool().acquire() as con:
        await con.execute(
            f"UPDATE runs SET {set_clause} WHERE id = $1",
            run_id, *values,
        )


async def delete_run(run_id: str):
    """Delete a run and cascade-delete its orthos and jobs."""
    async with pool().acquire() as con:
        await con.execute("DELETE FROM runs WHERE id = $1", run_id)


def _row_to_run(row: asyncpg.Record) -> dict:
    d = dict(row)
    if isinstance(d.get("params"), str):
        try:
            d["params"] = json.loads(d["params"])
        except Exception:
            d["params"] = {}
    for k in ("created_at", "finished_at", "started_at", "updated_at"):
        if isinstance(d.get(k), datetime):
            d[k] = d[k].astimezone(timezone.utc).isoformat()
    return d


# ═══════════════════════════════════════════════════════════════════════════
# Orthos (uploaded orthomosaics per run)
# ═══════════════════════════════════════════════════════════════════════════
async def create_ortho(
    run_id: str,
    original_filename: str,
    size_bytes: int,
    upload_order: int,
) -> str:
    """Insert an ortho record on upload. Returns the ortho id."""
    ortho_id = str(uuid.uuid4())
    async with pool().acquire() as con:
        await con.execute("""
            INSERT INTO orthos (id, run_id, original_filename, size_bytes, upload_order)
            VALUES ($1, $2, $3, $4, $5);
        """, ortho_id, run_id, original_filename, size_bytes, upload_order)
    return ortho_id


async def update_ortho(ortho_id: str, **fields):
    """Update arbitrary columns on an ortho."""
    if not fields:
        return
    # Convert date objects to string for asyncpg
    cols = list(fields.keys())
    set_clause = ", ".join(f"{c} = ${i+2}" for i, c in enumerate(cols))
    values = [fields[c] for c in cols]
    async with pool().acquire() as con:
        await con.execute(
            f"UPDATE orthos SET {set_clause} WHERE id = $1",
            ortho_id, *values,
        )


async def list_orthos(run_id: str) -> list[dict]:
    """List all orthos for a run, in upload order."""
    async with pool().acquire() as con:
        rows = await con.fetch(
            "SELECT * FROM orthos WHERE run_id = $1 ORDER BY upload_order",
            run_id,
        )
    return [_row_to_ortho(r) for r in rows]


async def get_ortho_by_filename(run_id: str, original_filename: str) -> dict | None:
    async with pool().acquire() as con:
        row = await con.fetchrow(
            "SELECT * FROM orthos WHERE run_id = $1 AND original_filename = $2",
            run_id, original_filename,
        )
    return _row_to_ortho(row) if row else None


async def count_orthos(run_id: str) -> int:
    async with pool().acquire() as con:
        return await con.fetchval("SELECT COUNT(*) FROM orthos WHERE run_id = $1", run_id)


async def sum_ortho_bytes(run_id: str) -> int:
    async with pool().acquire() as con:
        val = await con.fetchval("SELECT COALESCE(SUM(size_bytes), 0) FROM orthos WHERE run_id = $1", run_id)
    return int(val)


async def delete_ortho(ortho_id: str):
    async with pool().acquire() as con:
        await con.execute("DELETE FROM orthos WHERE id = $1", ortho_id)


def _row_to_ortho(row: asyncpg.Record) -> dict:
    d = dict(row)
    for k in ("created_at",):
        if isinstance(d.get(k), datetime):
            d[k] = d[k].astimezone(timezone.utc).isoformat()
    if isinstance(d.get("acquisition_date"), date):
        d["acquisition_date"] = d["acquisition_date"].isoformat()
    return d


# ═══════════════════════════════════════════════════════════════════════════
# Jobs (compute attempts per run)
# ═══════════════════════════════════════════════════════════════════════════
async def create_job(run_id: str, job_type: str = "pipeline", request_id: str | None = None) -> str:
    """Insert a job record when a pipeline starts. Returns the job id."""
    job_id = str(uuid.uuid4())
    async with pool().acquire() as con:
        await con.execute("""
            INSERT INTO jobs (id, run_id, type, state, request_id, started_at)
            VALUES ($1, $2, $3, 'RUNNING', $4, now());
        """, job_id, run_id, job_type, request_id)
    return job_id


async def update_job(job_id: str, **fields):
    if not fields:
        return
    cols = list(fields.keys())
    set_clause = ", ".join(f"{c} = ${i+2}" for i, c in enumerate(cols))
    values = [fields[c] for c in cols]
    async with pool().acquire() as con:
        await con.execute(
            f"UPDATE jobs SET {set_clause} WHERE id = $1",
            job_id, *values,
        )


async def get_latest_job(run_id: str) -> dict | None:
    async with pool().acquire() as con:
        row = await con.fetchrow(
            "SELECT * FROM jobs WHERE run_id = $1 ORDER BY started_at DESC LIMIT 1",
            run_id,
        )
    return _row_to_job(row) if row else None


async def list_jobs(run_id: str) -> list[dict]:
    async with pool().acquire() as con:
        rows = await con.fetch(
            "SELECT * FROM jobs WHERE run_id = $1 ORDER BY started_at DESC",
            run_id,
        )
    return [_row_to_job(r) for r in rows]


def _row_to_job(row: asyncpg.Record) -> dict:
    d = dict(row)
    for k in ("started_at", "finished_at"):
        if isinstance(d.get(k), datetime):
            d[k] = d[k].astimezone(timezone.utc).isoformat()
    return d
