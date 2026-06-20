# Deploying GENELLM webapp on Azure (via Docker Hub)

This directory builds a **self-contained, hybrid (GPU *or* CPU)** image — app
code, the embedding/model data, the fine-tuned weights, **and** the base model —
all baked in, and pushes it to Docker Hub so Azure (or any host) can pull the
single image and run it.

**No HuggingFace dependency.** The base model + tokenizer
(BiomedNLP-PubMedBERT) and the NLTK stopwords are copied out of the **local
cache in the running `genellmweb` container** at build time — never downloaded
from the hub. The image sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`, so
at runtime everything loads from disk and the container never contacts
HuggingFace. (The fine-tuned weights `state_dict_0.pth` are a local data file,
baked in with the rest of the data.)

**Hybrid GPU/CPU — automatic, the *same* image either way:**
- `app.py` already picks `cuda if torch.cuda.is_available() else cpu`.
- `torch==2.2.2+cu121` bundles its own CUDA runtime, so no `nvidia/cuda` base
  is needed. Run with `--gpus all` on a GPU host → GPU is used. Run with no
  GPU (App Service / plain ACI / CPU VM) → torch falls back to CPU.
- So this **does** run on cheap GPU-free Azure hosts (slower inference), and
  accelerates automatically wherever a GPU is present.

It does **not** modify the running `genellmweb` container, its image, `app.yml`,
or `code/app.py`. The existing HTTPS-on-this-server deployment is unaffected.

## What's different from the server image

| | Server image (`requirements/docker/`) | This Azure image (`deploy/azure/`) |
|---|---|---|
| Base | `nvidia/cuda:...-devel` (~6 GB) | `python:3.11-slim` (~4 GB total image) |
| GPU | required (`privileged`) | **optional** — GPU if `--gpus all`, else CPU |
| Code | bind-mounted from disk | **baked in** |
| Data (~1 GB) | bind-mounted from `/data/web_data` | **baked in** (only the 8 files app.py loads) |
| Base model | downloaded from HF at runtime | **baked in from local cache** (`*_OFFLINE=1`, never hits HF) |
| TLS | gunicorn serves HTTPS with self-signed `cert.pem`/`key.pem` | **plain HTTP** on :5000; TLS terminated upstream |
| Entrypoint | `python3 app.py` → re-execs gunicorn **with** cert | `gunicorn app:app` directly (no cert; `app.py` untouched) |
| Secrets | — | `.env`/`*.pem` excluded via `.dockerignore` |

## Build & push

```bash
docker login
IMAGE=<dockerhub-user>/genellmweb:1.0 ./deploy/azure/build_and_push.sh
```

- Image base is `nvidia/cuda` → the container needs a **GPU host** to run.
- To bake the full 3.7 GB data dir instead of just the used files:
  `INCLUDE_ALL_DATA=1 IMAGE=... ./deploy/azure/build_and_push.sh`

## TLS (Option B)

The container speaks plain HTTP. Put a front end in front to terminate public
TLS (it provisions/auto-renews a real trusted cert; no browser warnings, and no
private key ever lives in the image):

- **Azure Front Door** or **Application Gateway** → backend = container `:5000`.
- Or any reverse proxy (the existing UNM proxy works the same way).

## Run on Azure — pick a GPU target

The image needs an NVIDIA GPU. GPU quota for NC/ND-series VMs is **not** granted
by default — request it in the Azure portal first.

### Simplest: GPU VM + Docker

```bash
# on an Azure NC-series VM with the NVIDIA Container Toolkit installed:
docker login
docker pull <dockerhub-user>/genellmweb:1.0
docker run -d --restart=always --gpus all -p 5000:5000 \
  -e SMTP_USERNAME=... -e SMTP_PASSWORD=... \
  --name genellmweb <dockerhub-user>/genellmweb:1.0
```

### AKS (Kubernetes)

Use a GPU node pool + the NVIDIA device plugin; request `nvidia.com/gpu: 1` in
the pod spec, image = `<dockerhub-user>/genellmweb:1.0`, container port 5000,
front it with an Ingress/Application Gateway for TLS.

> Note: Azure Container Instances GPU SKUs are being retired / region-limited —
> not a reliable target anymore.

## Runtime env vars

| Var | Needed? | Purpose |
|---|---|---|
| `SMTP_USERNAME` | optional | `app.py` reads via `os.getenv`; only needed if email features are used |
| `SMTP_PASSWORD` | optional | as above |

Set these as Azure container env vars / app settings / a K8s Secret — never bake
them into the image.

## Notes

- `counter.txt` (hit counter) is baked in and written to the container's
  ephemeral layer; it resets on each redeploy. Harmless.
- The app makes outbound HTTPS calls to `g:Profiler` (gprofiler API); Azure
  containers have egress by default.
