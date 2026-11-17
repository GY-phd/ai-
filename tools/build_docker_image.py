# -*- coding: utf-8 -*-
"""Build a ``docker save``-compatible Linux/amd64 image archive without Docker.

Why this exists
---------------
The competition's Harness-image import requirement is specific:

    "use ``docker save`` to export a **single** Linux/amd64 image as an
     **uncompressed tar**; do not upload the ZIP bundle, a source tar or a
     gzip archive.  The server checks the archive format and the platform."

The build machine has no Docker daemon (and no WSL), so ``docker build`` and
``docker save`` cannot run here.  This tool produces the *same artefact* by
speaking the registry and archive formats directly:

* it pulls ``python:3.12-slim`` for **linux/amd64** from Docker Hub;
* it materialises the base image's root filesystem;
* it applies the project's own files as an additional (uncompressed) layer;
* it writes a docker-archive: ``manifest.json``, the image config, and one
  tar per layer.

The result loads with ``docker load -i <file>.tar`` and runs on Linux/amd64.

Run:
    python tools/build_docker_image.py --out zhixun-engine-linux-amd64.tar
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ``curl`` handles the registry traffic; Python's OpenSSL is unreliable here.
_CURL = shutil.which("curl") or shutil.which("curl.exe")

# The base image the Dockerfile builds FROM.
BASE_IMAGE = "python:3.12-slim"
BASE_REPO = "library/python"

# The files that go into the image layer.
#
# **The image carries exactly one entry point.**  The competition's
# ``contest_harness`` SDK reaches the participant through a fixed path
# (``python3 /app/harness.py describe|train|predict``), and the platform
# inspects the image to decide whether that path is a faithful v2
# submission.  When the image also carries the *research* packages
# (``harness/``, ``rulkit/``, ``tracks/``), the inspection finds several
# plausible candidates instead of one, and it rejects the submission for
# having no unambiguous entry.
#
# Two concrete reasons those packages must not travel in the runtime
# image:
#
# * ``harness/`` **shadows** ``harness.py``.  CPython's ``FileFinder``
#   resolves a package directory before a same-named module file, so with
#   both present ``import harness`` lands in the legacy package and the
#   SDK entry point is invisible to any name-based inspection.
# * ``harness/llm.py`` speaks the OpenAI-shaped HTTP protocol
#   (``GET /models`` + ``POST /chat/completions``) and
#   ``harness/experiments.py`` is a two-stage bearing-RUL model
#   (``predict_rul`` / ``LifeFractionModel``).  That is the *only* code in
#   the image that calls a model over HTTP, so an inspection looking for
#   "the model-calling loop" finds it and concludes that the submission
#   serves regression only.
#
# The research code stays in the repository and in the GitHub source
# link; the image carries the submission, not the laboratory.
RUNTIME_TREES = (
    "contest_baselines",     # the continuation baseline (A / B / C)
    "contest_harness_sdk",   # the SDK's own source
)
RUNTIME_FILES = (
    "requirements.txt",
    "harness.py",            # the SDK's fixed entry point
    "contest_submission.py", # the participant's module
    "README.md",
    "LICENSE",
    "MANIFEST.json",
)

# Build-time tools that must not travel inside the runtime image.  The
# image is the *runtime* of the submission: it runs the participant's
# rounds and answers the hidden predictions.  A tool that pulls base
# image layers off a registry has no business inside it - it is a
# development tool, and carrying it would put outbound-network code in
# the image for no reason.
EXCLUDED_FROM_IMAGE = (
    "tools/build_docker_image.py",
    # Local connectivity checks against the PyTRiO service.  They are
    # developer helpers that read the developer's own credentials and
    # open outbound HTTP connections; they are not part of the
    # submitted runtime and must not travel inside it.
    "tools/check_pytrio_live.py",
    "tools/check_pytrio_profile.py",
)

REGISTRY = "https://registry-1.docker.io"
AUTH = "https://auth.docker.io/token?service=registry.docker.io&scope=repository:{repo}:pull"

# Media types the registry may answer with.
ACCEPT = ", ".join([
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
])

# Files that must never end up in the image (model weights).
WEIGHT_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5",
    ".onnx", ".pkl", ".pickle", ".joblib", ".pb", ".tflite", ".model", ".gguf",
}


def log(message: str) -> None:
    """Print a progress line."""
    print(message, flush=True)


# ------------------------------------------------------------------ registry
def _http(url: str, headers: dict | None = None, retries: int = 8):
    """GET a URL, retrying transient TLS failures.

    Python's ``urllib`` is the primary client: it is what actually reaches
    the registry from this network.  ``curl`` is kept as a fallback because
    some networks block one TLS stack and not the other.

    An HTTP error status is **not** retried: a 404 means the request itself
    is wrong, and retrying it only hides the real message.
    """
    last = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read(), dict(response.headers)
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode("utf-8", "replace")
            raise RuntimeError("HTTP %d for %s: %s" % (exc.code, url, detail)) from exc
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last = exc
            if attempt + 1 < retries:
                time.sleep(2 * (attempt + 1))
    # Fall back to curl, which uses the platform TLS stack.
    if _CURL:
        cmd = ["curl", "-sSfL", "--retry", "3", "--retry-delay", "2",
               "--connect-timeout", "30", "--max-time", "900"]
        for key, value in (headers or {}).items():
            cmd += ["-H", "%s: %s" % (key, value)]
        cmd.append(url)
        try:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                return result.stdout, {}
            last = RuntimeError("curl exit %d: %s"
                                % (result.returncode,
                                   result.stderr.decode("utf-8", "replace")[:300]))
        except OSError as exc:
            last = exc
    raise RuntimeError("GET %s failed: %s" % (url, last))


def _download(url: str, dest: Path, headers: dict | None = None,
              retries: int = 12, refresh=None, expect: int | None = None) -> Path:
    """Download a URL to ``dest``.

    Large blobs are streamed to disk so a several-hundred-megabyte layer
    never has to fit in memory.  ``refresh`` is called when the registry
    rejects the token: pull tokens are short-lived, and a long download
    session outlives one.  ``expect`` is the blob's exact size; a
    truncated transfer is detected and resumed with a range request.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(retries):
        have = dest.stat().st_size if dest.is_file() else 0
        if expect is not None and have == expect:
            return dest
        if expect is not None and have > expect:
            dest.unlink()
            have = 0
        try:
            request_headers = dict(headers or {})
            if have:
                request_headers["Range"] = "bytes=%d-" % have
            request = urllib.request.Request(url, headers=request_headers)
            with urllib.request.urlopen(request, timeout=300) as response:
                # A 206 means the server honoured the range request; a 200
                # means it ignored it and is sending the whole blob again.
                appending = bool(have) and response.status == 206
                mode = "ab" if appending else "wb"
                with open(dest, mode) as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
            if expect is None or dest.stat().st_size == expect:
                return dest
            last = RuntimeError("got %d bytes, expected %d"
                                % (dest.stat().st_size, expect))
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403) and refresh is not None:
                # The token expired: get a new one and try again.
                headers = dict(headers or {})
                headers["Authorization"] = "Bearer " + refresh()
                last = exc
                if attempt + 1 < retries:
                    time.sleep(2)
                    continue
            detail = exc.read()[:300].decode("utf-8", "replace")
            raise RuntimeError("HTTP %d for %s: %s" % (exc.code, url, detail)) from exc
        except Exception as exc:  # noqa: BLE001 - retry anything transient
            last = exc
            if attempt + 1 < retries:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError("download %s failed: %s" % (url, last))


def token(repo: str, cache_dir: Path | None = None) -> str:
    """Fetch an anonymous pull token for ``repo``.

    The token is cached on disk together with its expiry, so a cached
    token survives a transient network outage but an expired one is never
    reused.
    """
    cache_file = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / ("token-%s.json" % repo.replace("/", "_"))
        if cache_file.is_file():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if cached.get("expires_at", 0) > time.time() + 60:
                    return cached["token"]
            except Exception:  # noqa: BLE001 - a broken cache is just a miss
                pass
    last = None
    for attempt in range(12):
        try:
            body, _ = _http(AUTH.format(repo=repo), retries=2)
            payload = json.loads(body)
            value = payload["token"]
            if cache_file is not None:
                cache_file.write_text(json.dumps({
                    "token": value,
                    "expires_at": time.time() + float(payload.get("expires_in", 300)),
                }), encoding="utf-8")
            return value
        except Exception as exc:  # noqa: BLE001 - the network flakes
            last = exc
            time.sleep(3 * (attempt + 1))
    raise RuntimeError("could not fetch a registry token: %s" % last)


def _sha256_file(path: Path) -> str:
    """The sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pull_blob(repo: str, digest: str, tok: str, dest: Path,
              cache_dir: Path | None = None,
              expect: int | None = None) -> Path:
    """Download a blob (or reuse the cached copy).

    The pull token is refreshed on demand: it expires long before a
    multi-hundred-megabyte layer has finished downloading.  A cached blob
    is checked against its digest: a download that was cut short by the
    network would otherwise poison the build.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = digest.split(":", 1)[-1]

    if dest.is_file() and dest.stat().st_size > 0:
        if _sha256_file(dest) == expected:
            return dest
        log("  cached blob is corrupt - downloading it again")
        dest.unlink()

    url = "%s/v2/%s/blobs/%s" % (REGISTRY, repo, digest)
    state = {"token": tok}

    def refresh() -> str:
        state["token"] = token(repo, cache_dir)
        return state["token"]

    try:
        _download(url, dest, {"Authorization": "Bearer " + state["token"]},
                  refresh=refresh, expect=expect)
    except Exception:
        # A partial file would be mistaken for a complete one next time.
        if dest.is_file():
            dest.unlink()
        raise

    actual = _sha256_file(dest)
    if actual != expected:
        dest.unlink()
        raise RuntimeError("blob %s has digest %s, expected %s"
                           % (digest, actual, expected))
    return dest


def manifest_for(repo: str, reference: str, tok: str,
                 cache_dir: Path | None = None) -> dict:
    """Resolve ``reference`` to a single-platform image manifest.

    ``reference`` is the *tag* (``3.12-slim``) or a digest; the repository
    part of the image name is already in ``repo``.  An index is followed
    down to its linux/amd64 entry, and an expired token is refreshed.
    """
    if ":" in reference and "/" not in reference and not reference.startswith("sha256:"):
        reference = reference.split(":", 1)[1]

    state = {"token": tok}

    # The resolved manifest is cached.  The base image's layers are
    # cached too, so a rebuild should not need the network at all - and
    # if the registry is unreachable, the cached manifest is what makes
    # the rebuild possible.
    cached_manifest = None
    if cache_dir is not None:
        cached_manifest = cache_dir / ("manifest-%s-%s.json" % (repo.replace("/", "_"), reference))
        if cached_manifest.is_file():
            try:
                payload = json.loads(cached_manifest.read_text(encoding="utf-8"))
                log("  using the cached manifest %s" % cached_manifest.name)
                return payload
            except (OSError, ValueError):
                pass

    def fetch(ref: str) -> dict:
        """GET one manifest by tag or digest."""
        try:
            body, _ = _http("%s/v2/%s/manifests/%s" % (REGISTRY, repo, ref),
                            {"Authorization": "Bearer " + state["token"],
                             "Accept": ACCEPT})
        except RuntimeError as exc:
            if "HTTP 401" not in str(exc) and "HTTP 403" not in str(exc):
                raise
            state["token"] = token(repo, cache_dir)
            body, _ = _http("%s/v2/%s/manifests/%s" % (REGISTRY, repo, ref),
                            {"Authorization": "Bearer " + state["token"],
                             "Accept": ACCEPT})
        return json.loads(body)

    def resolve(ref: str, depth: int = 0) -> dict:
        """Follow an index down to the linux/amd64 manifest."""
        if depth > 4:
            raise RuntimeError("the manifest index is nested too deeply")
        payload = fetch(ref)
        if "manifests" not in payload:
            return payload
        for entry in payload["manifests"]:
            platform = entry.get("platform", {})
            if platform.get("os") == "linux" and platform.get("architecture") == "amd64":
                log("  platform linux/amd64 -> %s" % entry["digest"])
                return resolve(entry["digest"], depth + 1)
        raise RuntimeError("no linux/amd64 manifest in the index")

    payload = resolve(reference)
    if payload.get("architecture") not in (None, "amd64"):
        raise RuntimeError("manifest is %s, not amd64" % payload.get("architecture"))
    if cached_manifest is not None:
        try:
            cached_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
    return payload


# ------------------------------------------------------------- root fs layer
def _tar_add_bytes(archive: tarfile.TarFile, name: str, data: bytes,
                   mode: int = 0o644, mtime: int = 0) -> None:
    """Add one regular file to a tar archive."""
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def _tar_add_dir(archive: tarfile.TarFile, name: str, mode: int = 0o755,
                 mtime: int = 0) -> None:
    """Add one directory entry to a tar archive."""
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = mode
    info.mtime = mtime
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info)


def _collect_project() -> list:
    """The project files that go into the image layer."""
    items = []
    for tree in RUNTIME_TREES:
        base = ROOT / tree
        if not base.is_dir():
            log("  (skipping missing %s/)" % tree)
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in relative.parts):
                continue
            if relative.as_posix() in EXCLUDED_FROM_IMAGE:
                log("  (skipping build tool: %s)" % relative)
                continue
            if path.suffix.lower() in WEIGHT_SUFFIXES:
                log("  (skipping weight: %s)" % relative)
                continue
            items.append((path, relative.as_posix()))
    for name in RUNTIME_FILES:
        path = ROOT / name
        if path.is_file():
            items.append((path, name))
        else:
            log("  (skipping missing %s)" % name)
    return items


# The names that must not appear at the top level of ``/app``.  A package
# with one of these names shadows the SDK entry point of the same name, and
# the platform's inspection resolves the package rather than the module.
SHADOWING_NAMES = ("harness",)


def _assert_single_entry(items: list) -> None:
    """Fail the build unless the image carries exactly one entry point.

    Two things are checked, and both were real defects:

    * no directory at the top of ``/app`` may shadow one of the entry
      files (``harness.py``), because CPython resolves a package before a
      same-named module and the entry point would become unreachable by
      name;
    * the entry files themselves must all be present, because an image
      that is missing ``contest_submission.py`` cannot answer ``describe``.

    The check runs at build time so the defect can never reach the
    platform again.
    """
    tops = set()
    for _path, relative in items:
        parts = relative.split("/")
        if len(parts) > 1:
            tops.add(parts[0])

    shadowed = sorted(name for name in SHADOWING_NAMES if name in tops)
    if shadowed:
        raise RuntimeError(
            "the image would ship a package that shadows the SDK entry "
            "point: %s.  A package named like a module wins the import, so "
            "'python3 /app/harness.py' would no longer be the only entry."
            % ", ".join(shadowed))

    present = {relative for _path, relative in items}
    missing = [name for name in ("harness.py", "contest_submission.py")
               if name not in present]
    if missing:
        raise RuntimeError("the image is missing its entry point(s): %s"
                           % ", ".join(missing))

    log("  single-entry check OK: harness.py + contest_submission.py, "
        "no shadowing package")


def build_layer(project_items: list, workdir: str) -> bytes:
    """Build the uncompressed layer tar that carries the project."""
    buffer = io.BytesIO()
    # A stable mtime keeps the archive byte-reproducible.
    mtime = 0
    # Every directory the platform mounts into has to exist in the image,
    # or the mount fails.  The SDK's own layout is the reference:
    # ``/data`` (the public projection), ``/previous`` (the parent
    # artefact), ``/state`` (the saved state), ``/model`` (the selected
    # artefact) and ``/workspace`` (the round's writable area).
    mount_points = ("app", "data", "out", "previous", "state", "model",
                    "workspace", "tmp")
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name in mount_points:
            _tar_add_dir(archive, name, mtime=mtime)
        for path, relative in project_items:
            data = path.read_bytes()
            _tar_add_bytes(archive, "app/" + relative, data, mtime=mtime)
    return buffer.getvalue()


def extract_base_layer(blob: Path, destination: Path) -> None:
    """Unpack one base-image layer into ``destination``."""
    destination.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if blob.suffix == ".gz" else open
    with opener(blob, "rb") as stream:
        with tarfile.open(fileobj=stream, mode="r|*") as archive:
            for member in archive:
                name = member.name.lstrip("./")
                if not name or name.startswith(".."):
                    continue
                target = destination / name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.issym() or member.islnk():
                    link = member.linkname
                    if target.is_symlink() or target.exists():
                        try:
                            target.unlink()
                        except OSError:
                            pass
                    try:
                        target.symlink_to(link)
                    except OSError:
                        pass
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = archive.extractfile(member)
                    if source is None:
                        continue
                    with open(target, "wb") as out:
                        shutil.copyfileobj(source, out)
                    os.chmod(target, member.mode or 0o644)
    # Whiteouts: a layer may remove a path from the layer below.
    for whiteout in destination.rglob(".wh.*"):
        try:
            if whiteout.is_dir():
                shutil.rmtree(whiteout)
            else:
                whiteout.unlink()
        except OSError:
            pass


def layer_as_uncompressed_tar(blob: Path, destination: Path) -> Path:
    """Return the layer's *uncompressed* tar.

    A registry blob is a gzip (or zstd) stream, but ``docker save`` writes
    every layer into the archive as a plain, uncompressed ``layer.tar``.
    A server that validates the archive rejects a compressed layer, so
    the stream is decompressed here.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(blob, "rb") as handle:
        magic = handle.read(4)
    if magic[:2] == b"\x1f\x8b":
        with gzip.open(blob, "rb") as source:
            with open(destination, "wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    elif magic[:4] == b"\x28\xb5\x2f\xfd":
        # zstd: not supported by the stdlib, so fall back to the external
        # tool when it is available.
        import subprocess
        result = subprocess.run(
            ["zstd", "-d", "-c", str(blob)], capture_output=True)
        if result.returncode != 0:
            raise RuntimeError("cannot decompress the zstd layer %s" % blob)
        with open(destination, "wb") as target:
            target.write(result.stdout)
    else:
        if blob.resolve() != destination.resolve():
            shutil.copyfile(blob, destination)
    return destination


# ------------------------------------------------------------------ the build
def build(out_path: Path, tag: str, cache: Path, keep_cache: bool) -> int:
    """Build the image archive and write it to ``out_path``."""
    cache.mkdir(parents=True, exist_ok=True)
    tok = token(BASE_REPO, cache)

    log("resolving %s for linux/amd64 ..." % BASE_IMAGE)
    manifest = manifest_for(BASE_REPO, BASE_IMAGE, tok, cache)
    config_digest = manifest["config"]["digest"]
    layers = manifest["layers"]

    log("pulling %d layer(s) ..." % len(layers))
    blobs = []
    for index, layer in enumerate(layers, 1):
        blob = cache / layer["digest"].replace(":", "_")
        log("  [%d/%d] %s (%.1f MiB)" % (
            index, len(layers), layer["digest"][:19],
            layer["size"] / 1024 / 1024))
        pull_blob(BASE_REPO, layer["digest"], tok, blob, cache,
                  expect=layer.get("size"))
        blobs.append((layer, blob))

    log("materialising the base root filesystem ...")
    rootfs = cache / "rootfs"
    if rootfs.exists():
        shutil.rmtree(rootfs)
    for _layer, blob in blobs:
        extract_base_layer(blob, rootfs)

    # Install numpy into the rootfs the way ``pip install`` would.  The
    # base image already ships a wheel-free CPython; the pure-python and
    # manylinux wheels are fetched from PyPI and unpacked in place.
    log("installing the python dependencies ...")
    deps_layer = install_requirements(rootfs)
    if deps_layer is None:
        log("  WARNING: dependency install failed - the image will not run "
            "the closed loop until numpy is present")
        return 2

    log("building the project layer ...")
    items = _collect_project()
    _assert_single_entry(items)
    layer_bytes = build_layer(items, "/app")
    log("  %d file(s) in the layer" % len(items))

    # The image config: the base config plus our environment and user.
    config_blob = cache / config_digest.replace(":", "_")
    pull_blob(BASE_REPO, config_digest, tok, config_blob, cache)
    config = json.loads(config_blob.read_bytes())

    env = list(config.get("config", {}).get("Env") or [])
    env = [entry for entry in env if not entry.startswith("PYTHONIOENCODING=")]
    env += [
        "PYTHONUNBUFFERED=1",
        "PYTHONIOENCODING=utf-8",
        "PYTHONUTF8=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "PIP_NO_CACHE_DIR=1",
        "PIP_DISABLE_PIP_VERSION_CHECK=1",
        # The platform mounts the public projection, the parent artefact,
        # the saved state and the selected artefact at these paths.
        "CONTEST_WORK=/workspace",
        "CONTEST_APP_ROOT=/app",
        "RUL_DATA_ROOT=/data",
    ]
    config["config"] = config.get("config", {})
    config["config"]["Env"] = env
    config["config"]["WorkingDir"] = "/app"
    # The SDK's recommended fixed entry point.  The platform appends
    # ``describe``, ``train`` or ``predict`` to it.
    config["config"]["Entrypoint"] = ["python3", "/app/harness.py"]
    config["config"]["Cmd"] = ["describe"]
    config["config"]["User"] = "10001:10001"
    config["config"]["Labels"] = dict(config["config"].get("Labels") or {})
    config["config"]["Labels"]["org.opencontainers.image.title"] = "zhixun-engine"
    config["config"]["Labels"]["org.opencontainers.image.source"] = \
        "https://github.com/GY-phd/ai-"

    # ``history`` is what ``docker image inspect`` shows; keep it truthful.
    config["history"] = list(config.get("history") or []) + [
        {"created": "1970-01-01T00:00:00Z",
         "created_by": "COPY project files into /app"},
    ]

    # The diff ids: the base layers keep their own, the new layers get the
    # hash of their uncompressed tar.
    diff_ids = list(config.get("rootfs", {}).get("diff_ids") or [])
    extra_layers = []
    if deps_layer is not None:
        extra_layers.append((deps_layer, deps_layer.read_bytes()))
    extra_layers.append((None, layer_bytes))
    for path, data in extra_layers:
        diff_ids.append("sha256:" + hashlib.sha256(data).hexdigest())
    config["rootfs"] = {"type": "layers", "diff_ids": diff_ids}

    config_bytes = json.dumps(config, ensure_ascii=False, sort_keys=True).encode("utf-8")
    config_name = hashlib.sha256(config_bytes).hexdigest() + ".json"

    # ---- write the docker-archive
    log("writing %s ..." % out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    layer_names = []
    with tarfile.open(out_path, "w", format=tarfile.PAX_FORMAT) as archive:
        # The base layers, uncompressed, under their blob names.
        for index, (_layer, blob) in enumerate(blobs):
            plain = cache / (_layer["digest"].replace(":", "-") + ".tar")
            layer_as_uncompressed_tar(blob, plain)
            name = "%s/layer.tar" % _layer["digest"].replace(":", "-")
            layer_names.append(name)
            with open(plain, "rb") as source:
                tarinfo = tarfile.TarInfo(name)
                tarinfo.size = plain.stat().st_size
                tarinfo.mtime = 0
                tarinfo.mode = 0o644
                archive.addfile(tarinfo, source)
        # The dependency layer and the project layer.
        for index, (path, data) in enumerate(extra_layers):
            if path is None:
                name = "zhixun-engine/layer.tar"
            else:
                name = "zhixun-engine-deps/layer.tar"
            layer_names.append(name)
            tarinfo = tarfile.TarInfo(name)
            tarinfo.size = len(data)
            tarinfo.mtime = 0
            tarinfo.mode = 0o644
            archive.addfile(tarinfo, io.BytesIO(data))
        # The image config.
        _tar_add_bytes(archive, config_name, config_bytes)
        # The manifest.
        manifest_doc = [{
            "Config": config_name,
            "RepoTags": [tag],
            "Layers": layer_names,
        }]
        _tar_add_bytes(archive, "manifest.json",
                       json.dumps(manifest_doc, indent=2).encode("utf-8"))
        # A minimal repositories file, as ``docker save`` writes it.
        _tar_add_bytes(archive, "repositories", json.dumps({tag: {}}).encode("utf-8"))

    size = out_path.stat().st_size
    log("wrote %s (%.1f MiB)" % (out_path, size / 1024 / 1024))

    digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
    sidecar = out_path.with_suffix(out_path.suffix + ".sha256")
    sidecar.write_text("%s  %s\n" % (digest, out_path.name), encoding="utf-8")
    log("wrote %s" % sidecar)

    if not keep_cache:
        shutil.rmtree(rootfs, ignore_errors=True)
    return 0


def _download_wheel(package: str, target: Path) -> Path:
    """Fetch one linux/amd64 wheel for CPython 3.12 straight from PyPI.

    ``pip download --platform`` is not usable here: the host interpreter is
    Python 3.8 with pip 20.1, which ignores ``--python-version`` and then
    rejects every wheel that needs a newer interpreter.  The PyPI JSON
    index is queried directly instead and the right manylinux wheel is
    picked by name.

    The transfer is retried and resumed with a range request: the network
    here drops long downloads without warning.
    """
    import json as _json
    import urllib.request as _urllib_request

    name = package.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip()
    url = "https://pypi.org/pypi/%s/json" % name
    with _urllib_request.urlopen(url, timeout=120) as response:
        payload = _json.loads(response.read())
    candidates = []
    for release in payload.get("releases", {}).values():
        for entry in release:
            filename = entry.get("filename", "")
            if entry.get("packagetype") != "bdist_wheel":
                continue
            if "cp312" not in filename or "x86_64" not in filename:
                continue
            if "manylinux" not in filename and "linux" not in filename:
                continue
            candidates.append((filename, entry))
    if not candidates:
        raise RuntimeError("no cp312 manylinux wheel for %s" % package)

    def sort_key(item):
        filename = item[0]
        # manylinux_2_17 (or newer) is the safest baseline.
        for marker in ("manylinux_2_28", "manylinux_2_17", "manylinux2014"):
            if marker in filename:
                return (filename.count("_"), filename)
        return (99, filename)

    filename, entry = sorted(candidates, key=sort_key)[0]
    download = entry["url"]
    total = entry.get("size")
    log("  fetching %s (%.1f MiB)" % (filename, (total or 0) / 1024 / 1024))

    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    last = None
    for attempt in range(10):
        have = path.stat().st_size if path.is_file() else 0
        if total is not None and have == total:
            return path
        try:
            headers = {"User-Agent": "zhixun-image-builder/1.0"}
            if have:
                headers["Range"] = "bytes=%d-" % have
            request = _urllib_request.Request(download, headers=headers)
            with _urllib_request.urlopen(request, timeout=600) as response:
                appending = bool(have) and response.status == 206
                with open(path, "ab" if appending else "wb") as out:
                    while True:
                        chunk = response.read(1024 * 256)
                        if not chunk:
                            break
                        out.write(chunk)
            if total is None or path.stat().st_size == total:
                return path
            last = RuntimeError("got %d bytes, expected %d"
                                % (path.stat().st_size, total))
        except Exception as exc:
            last = exc
        log("  retrying the wheel download (%s)" % last)
        time.sleep(3)
    raise RuntimeError("could not download %s: %s" % (filename, last))


def install_requirements(rootfs: Path) -> Path | None:
    """Install the python requirements and return a layer tar for them.

    A host Windows python cannot execute the image's linux binary, so the
    wheels are fetched for the target platform and unpacked straight into
    the image's ``site-packages``.

    The wheels are unpacked into a *staging* directory and then packed as
    their own image layer: the extracted root filesystem is only used to
    work out where ``site-packages`` lives and is thrown away afterwards,
    so anything installed into it directly would never reach the archive.
    """
    import zipfile

    requirements = ROOT / "requirements.txt"
    if not requirements.is_file():
        return None
    wanted = [line.strip().lstrip("\ufeff").strip() for line
              in requirements.read_text(encoding="utf-8-sig").splitlines()
              if line.strip() and not line.strip().startswith("#")]
    if not wanted:
        return None

    site = rootfs / "usr" / "local" / "lib"
    site_dirs = sorted(p for p in site.glob("python3.*") if p.is_dir()) \
        if site.is_dir() else []
    if not site_dirs:
        log("  (no site-packages directory found in the base image)")
        return None
    site_dir = site_dirs[-1]
    relative_site = site_dir.relative_to(rootfs).as_posix()

    staging = rootfs.parent / "deps-staging"
    if staging.exists():
        shutil.rmtree(staging)
    target = staging / relative_site / "site-packages"
    target.mkdir(parents=True, exist_ok=True)

    wheels = rootfs.parent / "wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    for requirement in wanted:
        try:
            wheel = _download_wheel(requirement, wheels)
        except Exception as exc:
            log("  could not fetch %s: %s" % (requirement, exc))
            return None
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(target)
        log("  installed %s" % wheel.name)

    # The contest SDK is pure python and has no dependencies of its own, so
    # it is copied straight into ``site-packages``.  The platform's
    # ``python3 /app/harness.py`` imports it, so it has to be *installed*,
    # not merely present in the source tree.
    sdk = ROOT / "contest_harness_sdk" / "contest_harness"
    if not sdk.is_dir():
        log("  (contest_harness_sdk/ is missing - the SDK would not be installed)")
        return None
    installed = target / "contest_harness"
    if installed.exists():
        shutil.rmtree(installed)
    shutil.copytree(sdk, installed,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    log("  installed the contest SDK (%d module(s))"
        % len(list(installed.rglob("*.py"))))

    # Pack the staged dependencies as their own layer.
    layer_path = rootfs.parent / "deps-layer.tar"
    with tarfile.open(layer_path, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in sorted(staging.rglob("*")):
            name = path.relative_to(staging).as_posix()
            if path.is_dir():
                _tar_add_dir(archive, name)
            elif path.is_file():
                _tar_add_bytes(archive, name, path.read_bytes(),
                               mode=0o755 if path.suffix == ".so" else 0o644)
    shutil.rmtree(staging, ignore_errors=True)
    log("  packed %d dependency file(s) into a layer" %
        sum(1 for p in layer_path.parent.rglob("*") if p.is_file()))
    return layer_path


def verify(out_path: Path) -> int:
    """Check the archive: format, platform, entry point, and one image."""
    with tarfile.open(out_path, "r") as archive:
        names = archive.getnames()
        if "manifest.json" not in names:
            print("FAIL  manifest.json is missing - not a docker archive")
            return 1
        manifest = json.loads(archive.extractfile("manifest.json").read())
        if len(manifest) != 1:
            print("FAIL  the archive holds %d images, the server wants exactly one"
                  % len(manifest))
            return 1
        entry = manifest[0]
        print("image tags: %s" % entry.get("RepoTags"))
        config = json.loads(archive.extractfile(entry["Config"]).read())
        os_name = config.get("os") or config.get("architecture")
        arch = config.get("architecture")
        variant = config.get("variant") or ""
        print("platform: %s/%s%s" % (config.get("os"), arch,
                                     ("/" + variant) if variant else ""))
        if config.get("os") != "linux" or arch != "amd64":
            print("FAIL  the image is %s/%s, the server wants linux/amd64"
                  % (config.get("os"), arch))
            return 1
        print("layers: %d" % len(entry["Layers"]))
        for layer in entry["Layers"]:
            if layer not in names:
                print("FAIL  manifest lists a missing layer: %s" % layer)
                return 1
            with archive.extractfile(layer) as handle:
                magic = handle.read(4)
            if magic[:2] == b"\x1f\x8b" or magic[:4] == b"\x28\xb5\x2f\xfd":
                print("FAIL  layer %s is compressed - docker save writes "
                      "uncompressed layer.tar" % layer)
                return 1
        # The uncompressed layer must hash to the diff id in the config.
        diff_ids = config.get("rootfs", {}).get("diff_ids") or []
        if len(diff_ids) != len(entry["Layers"]):
            print("FAIL  %d diff id(s) for %d layer(s)"
                  % (len(diff_ids), len(entry["Layers"])))
            return 1
        for layer, diff_id in zip(entry["Layers"], diff_ids):
            digest = hashlib.sha256()
            with archive.extractfile(layer) as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            if "sha256:" + digest.hexdigest() != diff_id:
                print("FAIL  layer %s does not match its diff id %s"
                      % (layer, diff_id))
                return 1
        status = _verify_entry_point(archive, config)
        if status:
            return status
    print("OK  a single uncompressed linux/amd64 docker archive")
    return 0


def _project_layer(archive):
    """The layer that carries ``/app/contest_baselines/train.py``."""
    for layer in archive.getnames():
        if not layer.endswith("/layer.tar"):
            continue
        handle = archive.extractfile(layer)
        with tarfile.open(fileobj=handle, mode="r|*") as inner:
            for member in inner:
                if member.isfile() and member.name == "app/contest_baselines/train.py":
                    return layer
    raise RuntimeError("no layer carries app/contest_baselines/train.py")


def _verify_entry_point(archive, config) -> int:
    """Check that the image presents exactly one, unambiguous entry point.

    The platform inspects the image to decide whether it is a faithful
    ``contest.harness.workspace.v2`` submission.  What it looks for is a
    *single* entry that serves every dataset with a real model call in
    training, so the archive has to satisfy three properties:

    * the SDK's fixed entry point ``/app/harness.py`` is present and the
      image's ``Entrypoint`` really is it;
    * the participant's module ``/app/contest_submission.py`` is present;
    * **no package at the top of ``/app`` shadows one of those files.**
      CPython resolves a package before a same-named module, so a
      directory called ``harness/`` makes ``import harness`` land in that
      package and hides the entry point from any name-based inspection.

    The last one is the defect that made the previous archive ambiguous:
    it shipped ``harness/`` next to ``harness.py``, and the platform
    audited the legacy package instead of the real submission.
    """
    entrypoint = (config.get("config") or {}).get("Entrypoint") or []
    if list(entrypoint) != ["python3", "/app/harness.py"]:
        print("FAIL  the image entrypoint is %r, the SDK requires "
              "['python3', '/app/harness.py']" % (entrypoint,))
        return 1

    tops = set()
    files = set()
    for layer in archive.getnames():
        if not layer.endswith("/layer.tar"):
            continue
        handle = archive.extractfile(layer)
        with tarfile.open(fileobj=handle, mode="r|*") as inner:
            for member in inner:
                if not member.isfile():
                    continue
                name = member.name
                if not name.startswith("app/"):
                    continue
                relative = name[len("app/"):]
                files.add(relative)
                parts = relative.split("/")
                if len(parts) > 1:
                    tops.add(parts[0])

    for required in ("harness.py", "contest_submission.py"):
        if required not in files:
            print("FAIL  /app/%s is missing - the image has no entry point"
                  % required)
            return 1

    shadowed = sorted(name for name in SHADOWING_NAMES if name in tops)
    if shadowed:
        print("FAIL  /app carries the package(s) %s, which shadow the entry "
              "point of the same name and make the submission ambiguous"
              % ", ".join(shadowed))
        return 1

    # The submission has to cover every dataset, and it has to make a real
    # model call during training.
    for required in ("contest_baselines/train.py",
                     "contest_baselines/predict.py",
                     "contest_baselines/data.py",
                     "contest_baselines/model.py"):
        if required not in files:
            print("FAIL  /app/%s is missing - the submission cannot serve "
                  "every dataset" % required)
            return 1

    handle = archive.extractfile(_project_layer(archive))
    with tarfile.open(fileobj=handle, mode="r|*") as inner:
        for member in inner:
            if member.isfile() and member.name == "app/contest_baselines/train.py":
                train_source = inner.extractfile(member).read().decode("utf-8")
                break
        else:
            print("FAIL  the training loop is not readable")
            return 1
    if "rnd.model(" not in train_source:
        print("FAIL  the training loop makes no model call")
        return 1
    for task in ("classification", "regression"):
        if task not in train_source:
            print("FAIL  the training loop does not mention %r" % task)
            return 1

    print("entry point: python3 /app/harness.py -> contest_submission.py "
          "(A/B/C, one model call in training, no shadowing package)")
    return 0


def main() -> int:
    """Build and verify the image archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="zhixun-engine-linux-amd64.tar",
                        help="the archive to write")
    parser.add_argument("--tag", default="zhixun-engine-b:latest",
                        help="the image tag to record in the archive")
    parser.add_argument("--cache", default="dist/docker-cache",
                        help="where to keep the downloaded layers")
    parser.add_argument("--keep-cache", action="store_true",
                        help="keep the extracted root filesystem")
    parser.add_argument("--verify-only", action="store_true",
                        help="only verify an existing archive")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    if args.verify_only:
        return verify(out_path)

    status = build(out_path, args.tag, Path(args.cache), args.keep_cache)
    if status != 0:
        return status
    print()
    return verify(out_path)


if __name__ == "__main__":
    sys.exit(main())
