# Offline, CPU-only image for the minimal Harness.
# Build:  docker build -t cpu-c-harness .
# Check:  docker run --rm cpu-c-harness doctor
# Demo:   docker run --rm -v "$PWD/runs:/app/runs" --entrypoint python \
#           cpu-c-harness tools/run_offline_demo.py --require-swanlab
# Run:    docker run --rm -v "$PWD/runs:/app/runs" -v "$PWD/stream:/data:ro" \
#           cpu-c-harness run --stream /data --workdir /app/runs/inference \
#           --config harness/configs/default.json --swanlab offline
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BASELINE_CPU_THREADS=1

WORKDIR /app

# Dependencies first so source edits do not invalidate the wheel layer.
COPY harness/requirements.txt harness/requirements-swanlab.txt /app/harness/
RUN python -m pip install --no-cache-dir \
        -r /app/harness/requirements.txt \
        -r /app/harness/requirements-swanlab.txt

COPY harness /app/harness
COPY tools /app/tools
COPY cpu-c-guide-20261110-v1 /app/cpu-c-guide-20261110-v1
COPY README.md /app/README.md

# The Harness is fully offline: no model API, no network, no GPU.
ENTRYPOINT ["python", "-m", "harness.cli"]
CMD ["doctor"]
