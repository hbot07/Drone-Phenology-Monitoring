# =============================================================================
# Drone Phenology Monitoring (DPM) — deps-only pipeline + dashboard image
#
# CoRE Stack cluster rule (docs.core-stack.org/server/cluster-docker-services,
# "Code, models, and data live on the host (mount, do not copy)"):
#   This image ships ONLY the OS packages and conda/pip environments.
#   Code, model weights, and all data/output live on the HOST and are
#   bind-mounted at runtime (see docker-compose.yml):
#       ./code   -> /app            (git checkout — update with `git pull` + restart)
#       ./models -> /app/models     (weights/checkpoints — never baked into the image)
#       ./data   -> /app/data       (inputs, caches, ALL compute output, logs)
#
# Build (from repo root, deps only — do NOT COPY app source in here):
#   docker build -t <registry>/<org>/dpm-pipeline:<version> .
#   docker tag <registry>/<org>/dpm-pipeline:<version> <registry>/<org>/dpm-pipeline:latest
#
# Push to GHCR or Docker Hub (cluster rule §5 — pinned tag, not only `latest`):
#   docker push <registry>/<org>/dpm-pipeline:<version>
#   docker push <registry>/<org>/dpm-pipeline:latest
#
# Rebuild only when THESE dependencies change (system libs, conda envs,
# CUDA/GDAL versions) — never for app code changes, which are a `git pull`.
# =============================================================================

FROM continuumio/miniconda3:latest

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System dependencies
# curl is required by both the Dockerfile HEALTHCHECK and the compose-level
# healthcheck — keep it even if the app itself doesn't use it directly.
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    build-essential \
    libgdal-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Install mamba for faster environment solving
# ---------------------------------------------------------------------------
RUN conda install -n base -c conda-forge mamba -y && \
    conda clean -afy

# ---------------------------------------------------------------------------
# dpm-tracking environment (FastAPI dashboard + phenology / tracking scripts)
# ---------------------------------------------------------------------------
RUN mamba create -y -n dpm-tracking \
    -c conda-forge \
    python=3.10 \
    "numpy=1.26.*" \
    pandas \
    geopandas \
    "rasterio=1.3.10" \
    shapely \
    "fiona=1.9.6" \
    "pyproj=3.6.1" \
    networkx \
    plotly \
    scikit-image \
    scikit-learn \
    opencv \
    rtree \
    scipy \
    matplotlib \
    flask \
    gdal \
    pip \
    && conda clean -afy

RUN conda run -n dpm-tracking pip install \
    fastapi \
    uvicorn \
    mercantile \
    pydantic \
    python-multipart \
    asyncpg \
    google-auth \
    PyJWT

# ---------------------------------------------------------------------------
# dpm-detectree environment (PyTorch + detectron2 + detectree2)
# ---------------------------------------------------------------------------
RUN mamba create -y -n dpm-detectree \
    -c pytorch -c nvidia -c conda-forge \
    python=3.10 \
    "numpy=1.26.*" \
    pandas \
    geopandas \
    "rasterio=1.3.10" \
    shapely \
    "fiona=1.9.6" \
    "pyproj=3.6.1" \
    matplotlib \
    "pytorch=2.3.1" \
    "torchvision=0.18.1" \
    "torchaudio=2.3.1" \
    "pytorch-cuda=12.1" \
    opencv \
    rtree \
    tqdm \
    pip \
    && conda clean -afy

# Fix setuptools — pkg_resources removed in setuptools 81+
# detectron2 setup.py imports it at build time
RUN conda run -n dpm-detectree pip install "setuptools<81" wheel

# detectron2 — pinned commit known to work with detectree2
# --no-build-isolation: ensures setup.py finds the venv's torch
RUN conda run -n dpm-detectree pip install --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git@e0ec4e189d438848521aee7926f9900e114229f5"

# detectree2
RUN conda run -n dpm-detectree pip install detectree2

# Pillow<10 required by detectree2
RUN conda run -n dpm-detectree pip install "Pillow==9.5.0"

# ---------------------------------------------------------------------------
# Make conda activate work inside container bash sessions
# ---------------------------------------------------------------------------
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /etc/bash.bashrc

# ---------------------------------------------------------------------------
# Working directory — CODE is mounted here at run time (./code -> /app).
# Nothing is COPY'd into the image: code, models, and data are never baked in.
# ---------------------------------------------------------------------------
WORKDIR /app

# ---------------------------------------------------------------------------
# Health check (CoRE Stack §6) — hits the /health endpoint added to server.py.
# curl is installed above for exactly this purpose.
# start_period gives uvicorn time to boot before the first probe fires.
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["/bin/bash"]
