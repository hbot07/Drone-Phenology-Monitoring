# Drone Phenology Monitoring

A repeated-drone-imagery pipeline for monitoring tree phenology at individual crown level, deployed as a Docker service on the [CoRE Stack](https://docs.core-stack.org/) Tower Services cluster.

Given a time series of georeferenced drone orthomosaics for a site, the pipeline detects individual tree crowns (Detectree2), tracks them across dates (graph-based matching), extracts crown-level phenology signals (GCC, RCC, texture, vegetation fractions), and classifies each crown's phenophase (deciduous / evergreen, leaf-on / leaf-off timing). The output is a STAC 1.1.0 Feature with the [table extension](https://stac-extensions.github.io/table/v1.2.0/schema.json), ready for catalog, GeoServer, or downstream satellite experiments.

> **Image:** `surryyansh/dpm-pipeline:1.0.0` (Docker Hub, deps-only)
> **DAG:** `drone_phenology_monitoring`
> **Version:** see [`VERSION`](VERSION)

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/hbot07/Drone-Phenology-Monitoring.git
cd Drone-Phenology-Monitoring

# 2. Configure
cp .env.example .env
# Edit .env — fill in Google client ID, secret key, DB password

# 3. Pull the deps-only image (pinned tag)
docker pull surryyansh/dpm-pipeline:1.0.0

# 4. Start
docker compose up -d

# 5. Health check
curl -f http://localhost:8090/health
# → {"status": "ok"}

# 6. Open the dashboard
open http://localhost:8090
```

---

## Clone Layout

```
Drone-Phenology-Monitoring/
├── src/
│   ├── flask_app_tracking/    # FastAPI server, auth, DB, tracking & phenology modules
│   │   ├── server.py          # Main app (backend + frontend, one container)
│   │   ├── auth.py            # Google SSO verification
│   │   ├── db.py              # PostgreSQL persistence (asyncpg)
│   │   ├── airflow_client.py  # Airflow trigger + poll glue
│   │   └── stac_utils.py      # STAC Item builder
│   ├── pipeline/              # Pipeline entry points
│   │   └── run_pipeline.sh    # Full pipeline orchestrator (6 steps)
│   └── notebooks/             # Exploratory / archive notebooks
├── deploy/
│   └── stacd/                 # STACD YAMLs (DAG, algorithm repo, dataset repo)
├── envs/                      # Conda environment specs
├── requirements/              # Companion pip requirements
├── scripts/                   # Setup helpers, satellite/RS scripts
├── misc/
│   ├── docs/                  # Operational guides (drone → orthomosaic → Detectree2)
│   └── ODM/                   # ODM/NodeODM runbooks
├── .env.example               # Every env var, commented — never commit .env
├── Dockerfile                 # Deps-only (CoRE Stack §2 — no app source)
├── docker-compose.yml         # Local dev + cluster-ready (three-mount block)
├── docker-push.yml            # GitHub Actions: build + push to Docker Hub
├── VERSION                    # Semver — image tags match this
├── outputs.yaml               # Output retention policy (§10)
└── STAC_OUTPUT_EXAMPLE.json   # Example STAC Item the pipeline returns
```

---

## Host Mounts

The image is **deps-only** — OS packages and conda/pip environments. Code, model weights, and all data live on the host and are bind-mounted at runtime.

| Host folder           | Container path  | Contents                                                                        |
| --------------------- | --------------- | ------------------------------------------------------------------------------- |
| **`code/`**   | `/app`        | Git checkout. Update with`git pull` + container restart.                      |
| **`models/`** | `/app/models` | Detectree2 weights (`.pth`). Default: `250312_flexi.pth`.                   |
| **`data/`**   | `/app/data`   | Uploads, run outputs, caches, logs at`data/logs/drone-phenology-monitoring/`. |

**Local dev shortcut:** `docker-compose.yml` mounts the whole repo (`.:/app`) for convenience. **Cluster deploy:** uncomment the three-mount block in `docker-compose.yml` and delete the single-mount line.

```yaml
# Cluster production — three separate mounts:
- ./code:/app
- ./models:/app/models
- ./data:/app/data
```

Rebuild the image **only** when dependencies change (Dockerfile / requirements). A code change is `git pull` + restart.

---

## Environment Variables (`.env`)

Copy `.env.example` → `.env` and fill in real values. **Never commit `.env`.**

### Core

| Variable            | Purpose                                          | Default                 |
| ------------------- | ------------------------------------------------ | ----------------------- |
| `DPM_IMAGE`       | Docker image to use                              | `dpm-pipeline:latest` |
| `DPM_PULL_POLICY` | `always` on cluster, `never` for local build | `never`               |

### Database (PostgreSQL)

| Variable            | Purpose                                       | Default          |
| ------------------- | --------------------------------------------- | ---------------- |
| `DPM_DB_NAME`     | Database name                                 | `dpm`          |
| `DPM_DB_USER`     | Database role                                 | `dpm`          |
| `DPM_DB_PASSWORD` | Role password                                 | *(change-me)*  |
| `DPM_DB_HOST`     | Hostname — local sidecar or central Postgres | `dpm-postgres` |
| `DPM_DB_PORT`     | Port                                          | `5432`         |

**Cluster:** point `DPM_DB_HOST` at the central Postgres instance; remove the `dpm-postgres` sidecar service from `docker-compose.yml`. Ask the server DBA for the database/role.

### Auth (Google SSO)

| Variable                     | Purpose                                                                                                          |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| `DPM_AUTH_ENABLED`         | Enable Google SSO (`true` / `false`)                                                                         |
| `DPM_GOOGLE_CLIENT_ID`     | OAuth 2.0 client ID from Google Cloud Console                                                                    |
| `DPM_GOOGLE_CLIENT_SECRET` | Client secret (only if server-side code exchange is used)                                                        |
| `DPM_SECRET_KEY`           | Signs the app's own JWT — random value,**not** the Google secret. Generate with `openssl rand -hex 32`. |
| `DPM_SESSION_HOURS`        | Session lifetime in hours (`24`)                                                                               |

### Compute Mode (Airflow Switch)

`AIRFLOW_API_BASE` is **the** switch — do not add a separate `COMPUTE_MODE` flag.

| `AIRFLOW_API_BASE`      | Behaviour                                          |
| ------------------------- | -------------------------------------------------- |
| **Empty / unset**   | Pipeline runs locally inside this container.       |
| **Set** (non-empty) | Dashboard triggers and polls Airflow via REST API. |

| Variable                                    | Purpose                                                                                          |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `AIRFLOW_API_BASE`                        | Airflow REST API base URL                                                                        |
| `AIRFLOW_DAG_ID`                          | `drone_phenology_monitoring`                                                                   |
| `AIRFLOW_USERNAME` / `AIRFLOW_PASSWORD` | Basic auth credentials                                                                           |
| `AIRFLOW_TOKEN`                           | Alternative bearer auth (set one, not both)                                                      |
| `CORESTACK_API_BASE`                      | LAN/Docker-network address of this backend, reachable from the Airflow worker (not`localhost`) |

### Frontend

| Variable             | Purpose                                                                | Default  |
| -------------------- | ---------------------------------------------------------------------- | -------- |
| `DPM_API_BASE_URL` | API base the frontend calls — relative so same-container deploy works | `/api` |

### Logging

| Variable         | Purpose                          | Default          |
| ---------------- | -------------------------------- | ---------------- |
| `LOG_LEVEL`    | `debug` / `info` / `error` | `info`         |
| `LOG_TIMEZONE` | Timezone for log timestamps      | `Asia/Kolkata` |

Logs are written to `data/logs/drone-phenology-monitoring/`. Tail them with:

```bash
# On the host
tail -f data/logs/drone-phenology-monitoring/*.log

# Via Docker
docker compose logs -f dashboard
```

### Proxy (IIT Delhi)

| Variable                         | Purpose                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| `HTTP_PROXY` / `HTTPS_PROXY` | Campus proxy (`http://proxy21.iitd.ac.in:3128`). Remove if not on campus.               |
| `NO_PROXY` / `no_proxy`      | Bypass list — keep`googleapis.com` entries or Google SSO token verification will fail. |

If `docker pull` times out behind the IIT Delhi proxy, set the [Docker daemon proxy](https://docs.core-stack.org/infra/tower-services/#docker-pull-behind-the-iit-delhi-proxy). Do **not** put `registry-1.docker.io` or `auth.docker.io` in `NO_PROXY`.

---

## Docker Image

The image is **deps-only** — it ships conda environments and system libraries, never application source.

```bash
# Pull (pinned tag)
docker pull surryyansh/dpm-pipeline:1.0.0

# Build locally (only when dependencies change)
docker build -t surryyansh/dpm-pipeline:1.0.0 .
docker push surryyansh/dpm-pipeline:1.0.0
```

The [GitHub Actions workflow](docker-push.yml) builds and pushes to Docker Hub on pushes that touch `Dockerfile`, `requirements/`, or `VERSION`, and on `v*` tags. Tags pushed: `:<VERSION>`, `:latest`, `:sha-<git-sha>`.

**Resource requirements:** GPU recommended for Detectree2 crown detection (CUDA 12.1). The `docker-compose.yml` reserves all available NVIDIA GPUs. CPU-only mode works for tracking/phenology steps if crowns are pre-detected.

### Health Check

The container has a built-in health check hitting `GET /health`:

```bash
curl -f http://localhost:8090/health
```

Returns `{"status": "ok"}` — no DB, no auth required. Docker uses this for container health probes.

---

## Airflow / STACD Integration

Three STACD YAMLs describe the pipeline job. Upload them in Airflow via **STACD → Initialize Workflow**.

| File                | Location                                 | What                                                                     |
| ------------------- | ---------------------------------------- | ------------------------------------------------------------------------ |
| DAG YAML            | `deploy/stacd/dpm_dag.yaml`            | Steps, dependencies, trigger params (`run_id` + pipeline tuning)       |
| Algorithm Repo YAML | `deploy/stacd/dpm_algorithm_repo.yaml` | API mode: Airflow POSTs to`POST /api/export-phenology` on this backend |
| Dataset Repo YAML   | `deploy/stacd/dpm_dataset_repo.yaml`   | Root input: orthomosaic series staged via the dashboard                  |

**DAG ID:** `drone_phenology_monitoring`

**Trigger:**

```bash
curl -u "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" \
  -H "Content-Type: application/json" \
  -X POST "$AIRFLOW_API_BASE/dags/drone_phenology_monitoring/dagRuns" \
  -d '{"conf": {"run_id": "<run-id>"}}'
```

**Poll:**

```bash
curl -u "$AIRFLOW_USERNAME:$AIRFLOW_PASSWORD" \
  "$AIRFLOW_API_BASE/dags/drone_phenology_monitoring/dagRuns/<dag_run_id>"
```

The UI proxies all Airflow calls through `/api/dag/run` and `/api/dag/status` — JavaScript never calls Airflow directly (no CORS; credentials stay off the page).

Remember to **unpause the DAG** before the first trigger.

---

## STAC Output

The pipeline returns a **STAC 1.1.0 Feature** with the [table extension](https://stac-extensions.github.io/table/v1.2.0/schema.json). See [`STAC_OUTPUT_EXAMPLE.json`](STAC_OUTPUT_EXAMPLE.json) for the full envelope.

Key fields in the output Item:

- `properties.table:columns` — every attribute of the phenology vector (chain_id, is_deciduous, deciduous_score, leaf-off/on timing, phenophase, etc.)
- `assets.data.href` — WFS endpoint to fetch the vector layer as GeoJSON
- `assets.style` — QGIS style file for phenophase visualization
- `assets.thumbnail` — preview PNG

API-mode response envelope: `{ "status", "asset_id", "stac_items": [ <STAC Item> ] }`.

---

## Output Retention

See [`outputs.yaml`](outputs.yaml) at the repo root. Summary:

| Path                     | Mode                   | TTL     | Description                            |
| ------------------------ | ---------------------- | ------- | -------------------------------------- |
| `data/runs/*/output/`  | `public`             | —      | Final phenology vectors and STAC Items |
| `data/runs/*/uploads/` | `private_persistent` | —      | User-uploaded orthomosaics             |
| `data/logs/`           | `delete`             | 30 days | Application logs                       |

---

## Pipeline Steps

The full pipeline runs six steps in sequence, orchestrated by `src/pipeline/run_pipeline.sh`:

1. **Crown Detection** (`01_crown_detection.py`) — Detectree2 inference on each orthomosaic, producing per-date crown polygons at multiple confidence thresholds.
2. **Crown Tracking** (`02_crown_tracking.py`) — Graph-based cross-date matching to build stable crown chains across the time series.
3. **Phenology Analysis** (`03_phenology_analysis.py`) — Extract crown-level signals (GCC, RCC, grayscale texture, Laplacian variance, vegetation fractions) and classify deciduousness.
4. **COG Tiling** (`04a_cog_tiling.py`) — Tile orthomosaics for the interactive viewer.
5. **Phenophase Classification** (`10_phenophase_classifier.py` + `12_apply_phenophase_to_geojson.py`) — Assign per-observation phenophase labels (leaf-on / leaf-off / transition) using a gradient-boosting classifier.
6. **Interactive Viewer** (`04b_interactive_viz.py`) — Generate a standalone HTML viewer with tracked crowns overlaid on tiled orthomosaics.

### Main Outputs

| Path                                                  | Description                                        |
| ----------------------------------------------------- | -------------------------------------------------- |
| `01_detectree/crowns_multithreshold/*.gpkg`         | Crown detections at multiple confidence thresholds |
| `02_tracking/consensus_crowns_complete_all.gpkg`    | Final tracked consensus crowns                     |
| `03_phenology/tree_master_geojson.geojson`          | Crown geometry + phenology signals                 |
| `03_phenology/tree_master_geojson_phenoclf.geojson` | With phenophase classification applied             |
| `04_viewer/index.html`                              | Standalone interactive viewer                      |

---

## Upgrade

```bash
cd code/                       # or wherever your checkout lives
git pull origin main
docker compose restart dashboard

# If dependencies changed (check VERSION / Dockerfile):
docker pull surryyansh/dpm-pipeline:<new-tag>
# Update DPM_IMAGE in .env
docker compose up -d dashboard
```

---

## Operational Guides

Detailed workflow guides are in [`misc/docs/`](misc/docs/README.md):

1. [Collecting drone imagery](misc/docs/01_collecting_drone_imagery.md)
2. [Building orthomosaics (WebODM)](misc/docs/02_building_orthomosaic_webodm.md)
3. [Running Detectree2](misc/docs/03_running_detectree2.md)
4. [Creating orthomosaic printouts with crown IDs](misc/docs/04_orthomosaic_printout_crown_ids.md)
5. [Preparing QField crown annotation projects](misc/docs/05_qfield_crown_annotation.md)

---

## Cluster Service Checklist

| #  | Item                                                                                | Status     |
| -- | ----------------------------------------------------------------------------------- | ---------- |
| 1  | Mount`code/`, `models/`, `data/`; compute output in `data/`                 | ✅         |
| 2  | `AIRFLOW_API_BASE` set → Airflow; empty → local compute                         | ✅         |
| 3  | Image pushed to Docker Hub (`surryyansh/dpm-pipeline:1.0.0`)                      | ✅         |
| 4  | Google SSO implemented (server-side token verification)                             | ✅         |
| 5  | Logs under`data/logs/drone-phenology-monitoring/`; `LOG_LEVEL` debug/info/error | ✅         |
| 6  | Backend + frontend in one Docker (FastAPI serves both)                              | ✅         |
| 7  | Frontend API base URL from`.env` (`DPM_API_BASE_URL`, default `/api`)         | ✅         |
| 8  | Architecture diagram                                                                | ✅         |
| 9  | Postgres via connection string to central server                                    | ✅         |
| 10 | `outputs.yaml` — public / private_persistent / delete                            | ✅         |
| 11 | Front page + demo video                                                             | 🔲 Pending |

---

## Front Page and Demo Video (§11)

The service URL (`/drone`) opens a landing page: project overview → Google SSO → dashboard. One demo video (project intro + create → run → FileBrowser tutorial) is embedded on the front page.

**Status:** Demo video pending recording and review.

---

## References

- [CoRE Stack — Cluster Docker Services](https://docs.core-stack.org/server/cluster-docker-services/)
- [CoRE Stack — Cluster Service Checklist](https://docs.core-stack.org/server/cluster-service-checklist/)
- [CoRE Stack — Tower Services](https://docs.core-stack.org/infra/tower-services/)
- [STACD YAML Guide](https://github.com/SaharshLaud/STACD_framework/blob/dev/README.md#11-writing-your-own-yaml-workflow)
- [STAC Specs](https://docs.core-stack.org/use-precomputed-data/stac-specs/)
- [Copy-this: Custom LULC](https://github.com/salil-123/Project)
