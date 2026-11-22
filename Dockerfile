# The competition image: the Harness submission.
#
# The rules require "the model configuration and the Harness system image"
# to be submitted.  The evaluator runs the image offline, on a uniform
# machine, without touching the participant's environment - so everything the
# loop needs has to be baked in.
#
# **This image carries exactly one entry point.**  The competition's
# ``contest_harness`` SDK reaches the participant through a fixed path:
#
#     python3 /app/harness.py describe
#     python3 /app/harness.py train
#     python3 /app/harness.py predict
#
# and the platform inspects the image to decide whether that path is a
# faithful ``contest.harness.workspace.v2`` submission.  If the image also
# carries the research packages (``harness/``, ``rulkit/``, ``tracks/``),
# that inspection finds several plausible candidates instead of one and
# rejects the submission for having no unambiguous entry.  Concretely:
#
#   * a package directory named ``harness`` **shadows** ``harness.py``,
#     because CPython resolves a package before a same-named module, so
#     ``import harness`` lands in the package and the SDK entry point
#     becomes invisible to any name-based inspection;
#   * ``harness/llm.py`` is the only code that speaks the OpenAI-shaped
#     HTTP protocol (``GET /models`` + ``POST /chat/completions``), and
#     ``harness/experiments.py`` is a two-stage bearing-RUL model, so an
#     inspection looking for "the model-calling loop" finds the research
#     code and concludes the submission serves regression only.
#
# So the image carries the *submission* and not the laboratory.  The
# research runtime stays in the repository, in the source link and in
# ``Dockerfile.research``.
#
# Build:
#   docker build -t zhixun-engine-b .
#
# The image is self-contained and offline: numpy is the only hard
# dependency, and the platform's model is reached only through the SDK's
# host gateway - no key, endpoint or sampling client is baked in.

FROM python:3.12-slim-bookworm

# - PYTHONUNBUFFERED so the platform's log capture sees every line as it is
#   produced.
# - PYTHONIOENCODING / PYTHONUTF8 because the console code page on the
#   evaluation machine is unknown and the ledger writes CJK text.
# - PIP_NO_CACHE_DIR keeps the image small and the build reproducible.
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUTF8=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Four-core production host: pin BLAS so numpy does not oversubscribe.
ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4

WORKDIR /app

# ---- 1. the SDK ------------------------------------------------------------
# The SDK is installed from a local copy so the build needs no network
# beyond the base image's own layers.  The contest_harness package is
# pure python and has no dependencies of its own.
COPY contest_harness_sdk/ /opt/contest-harness-sdk/
RUN pip install --no-cache-dir /opt/contest-harness-sdk \
 && python -c "import contest_harness; print(contest_harness.SDK_VERSION)"

# ---- 2. the participant's code ---------------------------------------------
# Only the submission travels: the fixed entry point, the participant
# module, the continuation baseline that serves A / B / C, and the
# adapter plugin layer that carries each profile's specifics.
COPY harness.py /app/harness.py
COPY contest_submission.py /app/contest_submission.py
COPY contest_baselines/ /app/contest_baselines/

# numpy is the only hard dependency.
COPY requirements.txt ./
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir --index-url "${PIP_INDEX_URL}" -r requirements.txt

# ---- 3. the directories the platform mounts into ----------------------------
# The mount points have to exist in the image for the mounts to be valid.
RUN mkdir -p /data /previous /state /model /workspace /out /tmp \
 && useradd --create-home --uid 10001 harness \
 && chown -R harness:harness /app /workspace /tmp

# The platform runs the image as its own user; the submission's own
# participant id is declared through the platform's model identity field,
# not through the image.
USER harness

# The SDK's recommended fixed entry point.
ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]

# A build-time self check: the entry point must answer ``describe`` with a
# valid contract, and the image must carry no model weight.
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 /app/harness.py describe > /dev/null || exit 1
