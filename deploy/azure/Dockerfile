# Self-contained, HYBRID GENELLM webapp image for pulling-and-deploying on Azure
# (or any host). Bakes in the app code, the embedding/model data, AND the
# HuggingFace base model -- so it runs offline with no bind mounts.
#
# GPU/CPU is automatic, no app change:
#   * app.py already picks `cuda if torch.cuda.is_available() else cpu`.
#   * torch==2.2.2+cu121 bundles its own CUDA runtime, so we do NOT need an
#     nvidia/cuda base. On a GPU host, run with `--gpus all` and the NVIDIA
#     Container Toolkit injects the driver -> GPU is used. On a CPU-only host,
#     torch reports cuda unavailable -> it transparently falls back to CPU.
# This keeps the image ~4 GB (vs ~9 GB on a CUDA-devel base) and lets the SAME
# image run on cheap GPU-free Azure hosts and on GPU hosts alike.
#
# TLS (Option B): serves PLAIN HTTP on :5000; TLS is terminated upstream
# (Azure Front Door / Application Gateway / reverse proxy). cert.pem/key.pem
# and .env are excluded via .dockerignore -- no secrets in the image. We run
# gunicorn directly (not `python3 app.py`), so app.py's __main__/TLS path never
# runs and app.py stays unmodified.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# libgomp1: OpenMP runtime needed by torch / scikit-learn / scipy
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (same pinned set the server image uses; torch resolves to
# the +cu121 wheel from PyPI, which runs on both GPU and CPU)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Bake in the base model + tokenizer (BiomedNLP-PubMedBERT) and NLTK stopwords
# from the LOCAL cache (extracted from the running container by
# build_and_push.sh) -- NOT downloaded from HuggingFace. With *_OFFLINE=1 above,
# the app loads these purely from disk and never contacts the hub.
COPY hf_cache  /root/.cache/huggingface
COPY nltk_data /root/nltk_data

# Bake in the app code and the data files app.py loads at import time
# (incl. the fine-tuned weights state_dict_0.pth, already a local file)
COPY code /home/tailab/genellm_backend/code
COPY data /home/tailab/data

WORKDIR /home/tailab/genellm_backend/code

EXPOSE 5000

# Plain HTTP. gunicorn imports app:app, so app.py's __main__/TLS path is skipped.
ENTRYPOINT ["gunicorn", \
    "--worker-class", "gthread", \
    "--workers", "1", \
    "--threads", "8", \
    "--timeout", "120", \
    "--graceful-timeout", "30", \
    "--bind", "0.0.0.0:5000", \
    "--access-logfile", "-", \
    "--error-logfile", "-", \
    "app:app"]
