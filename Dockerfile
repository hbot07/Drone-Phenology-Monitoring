# =============================================================================
# Drone Phenology Monitoring — pipeline + dashboard image
#
# Build (from repo root):
#   docker build -t dpm-pipeline:latest .
#
# The project directory is mounted at /workspace at run time — nothing is
# copied into the image. Code, models and data all live on the host.
# =============================================================================

FROM continuumio/miniconda3:latest

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------------------------------------------------------------------------
# System dependencies
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
# Working directory — project is mounted at /workspace at run time
# Code, models and data are never copied into the image.
# ---------------------------------------------------------------------------
WORKDIR /workspace

CMD ["/bin/bash"]
