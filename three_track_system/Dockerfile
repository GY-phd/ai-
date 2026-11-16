# Harness system image for the competition evaluator.
#
# The rules require "the model configuration and the Harness system image"
# to be submitted.  The evaluator runs the image offline, on a uniform
# machine, without touching the participant's environment - so everything the
# loop needs has to be baked in.
#
# Build:
#   docker build -t zhixun-engine-b .
#
# Run (the evaluator's shape: offline, no intervention, bounded time):
#   docker run --rm --network none \
#     -v /path/to/official-data:/data:ro \
#     -v /path/to/output:/out \
#     zhixun-engine-b
#
# The image entrypoint runs the closed loop unattended and writes the
# artefacts to /out.  It never asks a question and never waits for input.

FROM python:3.12-slim

# - PYTHONUNBUFFERED so the evaluator's log capture sees every line as it is
#   produced, not when the buffer happens to flush.
# - PYTHONIOENCODING=utf-8 because the console code page on the evaluation
#   machine is unknown and the logger writes CJK text.
# - PIP_NO_CACHE_DIR keeps the image small and the build reproducible.
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# numpy is the only hard dependency.  swanlab is deliberately *not* installed:
# the evaluation environment is offline, the logger already writes the
# SwanLab-shaped stream locally, and a cloud client would only add a failure
# mode to a run that must never fail.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY harness/ ./harness/
COPY rulkit/ ./rulkit/
COPY tracks/ ./tracks/
COPY tools/ ./tools/
COPY research/ ./research/
COPY tests/ ./tests/
COPY evidence/ ./evidence/
COPY docs/ ./docs/
COPY runs/ ./runs/
COPY figureuse/ ./figureuse/
COPY run_harness.sh run_all.ps1 run_harness.ps1 ./
COPY README.md LICENSE MANIFEST.json ./
COPY "智训引擎-系统总思路.md" "基岩智序-完整调试过程.md" ./

# The data pack is mounted at runtime; the directory has to exist in the image
# for the mount point to be valid.
RUN mkdir -p /data /out \
 && useradd --create-home --uid 10001 harness \
 && chown -R harness:harness /app /out
USER harness

ENV RUL_DATA_ROOT=/data \
    HARNESS_WORKSPACE=/out/harness \
    HARNESS_MAX_EXPERIMENTS=6 \
    HARNESS_MAX_WALL_SECONDS=2400 \
    HARNESS_LLM_MODE=auto

# Fail the build if the image would ever ship a weight file.
RUN python tools/verify_repo.py --dir /app

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD test -f /out/harness/run_report.json || exit 1

ENTRYPOINT ["python", "-m", "harness.run"]
CMD ["--workspace", "/out/harness", \
     "--batch-size", "200", \
     "--max-experiments", "6", \
     "--llm-mode", "auto"]
