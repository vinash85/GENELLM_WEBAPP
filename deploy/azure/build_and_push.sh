#!/usr/bin/env bash
# Build a SELF-CONTAINED GENELLM webapp image (app code + data baked in) and
# push it to Docker Hub, so it can be pulled and deployed on Azure as a single
# image.
#
# This does NOT touch the running `genellmweb` container, its image, app.yml,
# or code/app.py. It assembles an isolated build context under $STAGE and
# builds from there.
#
# Usage:
#   docker login                      # once
#   IMAGE=<dockerhub-user>/genellmweb:1.0 ./deploy/azure/build_and_push.sh
#
# Optional env:
#   DATA_SRC=/data/web_data           # where the data csv/pth files live
#   STAGE=/data/azure_build           # scratch build context (same FS as DATA_SRC)
#   INCLUDE_ALL_DATA=1                # bake EVERY file in DATA_SRC (3.7 GB), not just the 8 used
set -euo pipefail

IMAGE="${IMAGE:?Set IMAGE=<dockerhub-user>/genellmweb:<tag>, e.g. IMAGE=myuser/genellmweb:1.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_SRC="${DATA_SRC:-/data/web_data}"
STAGE="${STAGE:-/data/azure_build}"
INCLUDE_ALL_DATA="${INCLUDE_ALL_DATA:-0}"
# Container to pull the LOCAL base-model + NLTK caches from (no HuggingFace download)
MODEL_CACHE_CONTAINER="${MODEL_CACHE_CONTAINER:-genellmweb}"
HERE="$REPO_ROOT/deploy/azure"

echo ">> repo:   $REPO_ROOT"
echo ">> data:   $DATA_SRC"
echo ">> stage:  $STAGE"
echo ">> image:  $IMAGE"

[ -d "$DATA_SRC" ] || { echo "!! DATA_SRC not found: $DATA_SRC" >&2; exit 1; }

# --- assemble an isolated build context ---
rm -rf "$STAGE"
mkdir -p "$STAGE/code" "$STAGE/data"

cp "$HERE/Dockerfile"      "$STAGE/Dockerfile"
cp "$HERE/.dockerignore"   "$STAGE/.dockerignore"
cp "$REPO_ROOT/requirements/docker/requirements/requirements.txt" "$STAGE/requirements.txt"

# app code (cert.pem/key.pem/__pycache__ are dropped by .dockerignore at build)
cp -a "$REPO_ROOT/code/." "$STAGE/code/"

# base model + tokenizer + NLTK data: copy the LOCAL caches out of the running
# container (NOT a HuggingFace download). Requires $MODEL_CACHE_CONTAINER to exist.
if ! docker inspect "$MODEL_CACHE_CONTAINER" >/dev/null 2>&1; then
    echo "!! container '$MODEL_CACHE_CONTAINER' not found; cannot extract the local model cache." >&2
    echo "   Set MODEL_CACHE_CONTAINER=<name> to a container holding /root/.cache/huggingface." >&2
    exit 1
fi
echo ">> extracting local model + nltk caches from container '$MODEL_CACHE_CONTAINER'"
docker cp "$MODEL_CACHE_CONTAINER:/root/.cache/huggingface" "$STAGE/hf_cache"
docker cp "$MODEL_CACHE_CONTAINER:/root/nltk_data"          "$STAGE/nltk_data"

# data: real copy into the build context -- only the files app.py loads, unless
# INCLUDE_ALL_DATA=1. (Hardlinks aren't usable here: the data files are owned by
# another user and protected_hardlinks blocks linking files we don't own.)
if [ "$INCLUDE_ALL_DATA" = "1" ]; then
    echo ">> baking ALL data files (copying ~3.7 GB)"
    cp -a "$DATA_SRC/." "$STAGE/data/"
else
    echo ">> baking only the files app.py loads (see data_files.txt)"
    while IFS= read -r f; do
        case "$f" in ''|\#*) continue ;; esac
        [ -e "$DATA_SRC/$f" ] || { echo "!! missing data file: $DATA_SRC/$f" >&2; exit 1; }
        cp -a "$DATA_SRC/$f" "$STAGE/data/$f"
    done < "$HERE/data_files.txt"
fi

echo ">> staged data ($(du -sh --apparent-size "$STAGE/data" | cut -f1)):"
ls -la "$STAGE/data"

# --- build & push ---
docker build -t "$IMAGE" "$STAGE"
docker push "$IMAGE"

echo ">> done: pushed $IMAGE"
echo ">> staging left at $STAGE (~1 GB). 'rm -rf $STAGE' to reclaim."
