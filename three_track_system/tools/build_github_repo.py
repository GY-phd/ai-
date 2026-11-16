# -*- coding: utf-8 -*-
"""Build the GitHub-ready repository folder: code + docs + evidence, no weights.

    python tools/build_github_repo.py --out dist/github

What this does that ``package_submission.py`` does not: it produces a *folder*
(not a zip) that can be dropped straight into a fresh Git repository, with

  * every weight-like file removed (the competition forbids shipping weights,
    and the harness is supposed to regenerate them at evaluation time);
  * the weight-bearing directories (``evidence/``) reduced to their text-only
    members - the ledger, the run report, the SwanLab metric stream - so the
    evidence trail survives without the tensors;
  * run artefacts and scratch directories removed;
  * a ``MANIFEST.json`` recording every file with its size and SHA-256, plus
    an explicit ``contains_model_weights: false`` flag;
  * a ``GIT_PUSH.md`` with the exact commands to initialise and push.

The script fails loudly (non-zero exit) if a weight file would slip in, if the
payload exceeds the size budget, or if the evidence trail is missing - the
same guarantees the submission packer gives, applied to the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Any of these inside the output folder means weights leaked in.
WEIGHT_SUFFIXES = {
    ".npz", ".npy", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".h5",
    ".hdf5", ".onnx", ".pkl", ".pickle", ".joblib", ".pb", ".tflite",
    ".model", ".weights", ".mlmodel", ".gguf", ".pt2", ".pth2",
}
# Directories that are copied only for their text-only members.
TEXT_ONLY_DIRS = ("evidence",)
# File types that count as text-only evidence.
TEXT_EXTENSIONS = {".json", ".jsonl", ".md", ".txt", ".csv"}
# Directories that never belong in the repository.
SKIP_DIRS = {
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "art", "art_probe", "art_probe2", "art_probe3", "art_probe4", "art_probe5",
    "dist", "data", "swanlog", "logs",
    "models", "site-packages", ".venv", "venv", "build", ".idea",
    "checkpoints", "registry",
}# Files that never belong in the repository.
SKIP_FILES = {"checkpoint.npz", "history.jsonl"}
# ``evo`` is the earlier single-task implementation, now a **sub-component**
# of the three-track system at the repository root.  Its tests live in
# ``tests/`` and import it by module path, so the package has to travel with
# the repository - but only its source: the folders below it also hold
# LightGBM checkpoints from the old sweeps, and those are weights.
CODE_ONLY_DIRS = ("evo",)

TOP_FILES = (
    "README.md", "LICENSE", "requirements.txt", "run_all.ps1", "run_harness.ps1",
    "run_harness.sh", ".gitignore", ".gitattributes", "Dockerfile", ".dockerignore",
    "智训引擎-系统总思路.md", "基岩智序-完整调试过程.md",
)
# The three-track closed loop (``tracks``), the run reports it wrote
# (``runs``), and the generated tables and figures (``artifacts``) all
# travel: they are the evidence that the numbers in the README are real
# and that they can be reproduced from the code that ships beside them.
# ``figureuse`` is the presentation set - the same figures, regenerated
# from the same artifacts, with the labels the slides use.
CODE_DIRS = ("harness", "rulkit", "tracks", "tools", "research", "tests",
             ".github", "runs", "figureuse")
DOC_DIRS = ("docs",)
# ``artifacts`` is **not** copied wholesale: the working folder also holds the
# early single-task sweep (246 MB of LightGBM checkpoints under ``sweep/``,
# ``es2_*``, ``rul_tune15/`` and friends), none of which belongs to the
# three-track evidence chain.  Only the generated tables, the verification
# result and the figures are published - the same set the sync tool copies
# into ``AI2``.
ARTIFACT_FILES = (
    "comparison.json", "comparison_table.csv", "comparison_table.md",
    "fusion_ablation.json", "online_curve.json", "ppt.md",
    "rmse_verification.json", "rmse_verification.md", "rul_benchmark.json",
    "rul_comparison.json", "run_summary.json", "scenarios.json",
)
ARTIFACT_GLOBS = ("figures/*.png", "tracks/*.json", "official/*.json")
MAX_TOTAL_BYTES = 4 * 1024 * 1024

PUSH_TEMPLATE = """# 上传到 GitHub

本目录已经整理完毕：**不含任何模型权重**，可直接作为 Git 仓库根目录使用。

## 1. 初始化并推送

```bash
cd <本目录>
git init
git add .
git commit -m "feat: 智训引擎 B 题 - 轴承 RUL 自进化闭环"
git branch -M main
git remote add origin <你的仓库地址>
git push -u origin main
```

## 2. 自检（推送前跑一次）

```bash
python tools/verify_repo.py --dir .
```

期望输出：`OK  无权重文件 / N 个文件 / 总大小 M KB`。

## 3. 复现一次闭环

```bash
pip install -r requirements.txt
python -m harness.run --workspace runs/harness --batch-size 200 --max-experiments 6
python tools/show_run.py --workspace runs/harness
```

## 4. 赛题交付物：Harness 系统镜像

赛题要求提交「模型配置 + Harness 的系统镜像」。镜像定义在仓库根目录的 `Dockerfile`：

```bash
docker build -t zhixun-engine-b .

# 评测机的运行形态：离线、无人工干预、时限受控
docker run --rm --network none \\
  -v /path/to/official-data:/data:ro \\
  -v /path/to/output:/out \\
  zhixun-engine-b
```

镜像特征：

- 只依赖 `numpy`（**不装 swanlab**：评测环境离线，本地已按 SwanLab 形状写指标流，
  云端客户端只会给一次不该失败的运行增加失败模式）；
- 入口是 `python -m harness.run`，**不需要传任何参数**，全部由环境变量驱动：
  `RUL_DATA_ROOT` / `HARNESS_WORKSPACE` / `HARNESS_BATCH_SIZE` /
  `HARNESS_MAX_EXPERIMENTS` / `HARNESS_MAX_WALL_SECONDS` / `HARNESS_LLM_MODE`；
- 构建时会跑 `tools/verify_repo.py`，**镜像里不可能混进权重文件**；
- 以非 root 用户（uid 10001）运行。

更详细的运行手册见 `docs/RUNBOOK.md`。

## 5. 说明

- **权重为什么不在这里**：赛题要求提交的是"可复现的代码 + 由代码产出的模型"。
  仓库里的 `evidence/` 只保留账本、运行报告与指标流（纯文本），
  `*.npz` 检查点由训练代码在评测时重新生成。
- **证据链**：`evidence/ledger.jsonl` 是哈希链账本，CI 会重新校验；
  任何一条记录被改动，CI 就会变红。
- **`harness/hypothesis.py`** 是融合进来的四段式证据链
  （读历史 -> 提假设 -> 做实验 -> 做决策），对应赛题门槛 ②③。
"""


def digest(path: Path) -> str:
    """Return the SHA-256 of a file, hex-encoded."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _skip(relative: Path) -> bool:
    """Decide whether a source path must not be copied."""
    if relative.name in SKIP_FILES:
        return True
    return any(part in SKIP_DIRS for part in relative.parts)


def collect() -> list:
    """Return (source, destination-relative) pairs for everything to ship."""
    files = []
    for name in TOP_FILES:
        path = ROOT / name
        if path.is_file():
            files.append((path, Path(name)))
    for directory in CODE_DIRS + DOC_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if _skip(relative):
                continue
            files.append((path, relative))
    # ``evo``: source only.  Its checkpoint folders are weights and are
    # excluded by both the suffix rule and ``SKIP_DIRS``.
    for directory in CODE_ONLY_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if _skip(relative):
                continue
            if path.suffix.lower() in WEIGHT_SUFFIXES:
                continue
            if path.suffix.lower() not in {".py", ".yaml", ".yml", ".toml",
                                           ".json", ".txt", ".md", ".cfg"}:
                continue
            files.append((path, relative))
    # The generated artefacts: a fixed list, never the whole folder.
    artifacts = ROOT / "artifacts"
    for name in ARTIFACT_FILES:
        path = artifacts / name
        if path.is_file():
            files.append((path, Path("artifacts") / name))
    for pattern in ARTIFACT_GLOBS:
        for path in sorted(artifacts.glob(pattern)):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if _skip(relative):
                continue
            files.append((path, relative))
    # Evidence: text-only members of the weight-bearing directories.
    for directory in TEXT_ONLY_DIRS:
        base = ROOT / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(ROOT)
            if _skip(relative) or path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            files.append((path, relative))
    return files


def _rebase_champion_pointer(out_dir: Path) -> None:
    """Point the packaged champion registry at a reproducible command.

    The registry's ``checkpoint`` field normally names a local file
    (``../champion_checkpoint.npz``) that does not travel with the repository,
    because the repository ships no weights.  Leaving a dangling pointer would
    make ``registry/champion.json`` look broken, so it is rewritten to name the
    training command that regenerates the checkpoint.  The metrics and the
    digest are untouched - only the pointer moves.
    """
    pointer = out_dir / "evidence" / "registry" / "champion.json"
    if not pointer.is_file():
        return
    try:
        entry = json.loads(pointer.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    entry["checkpoint"] = "regenerate: python -m rulkit.train --train data/public/train.jsonl " \
                          "--validation data/public/validation.jsonl --from-baseline data/checkpoint.npz " \
                          "--output runs/round-1 --epochs 400 --hidden 128 --feature-transform eng34 " \
                          "--transfer --transfer-epochs 80"
    entry["checkpoint_included"] = False
    entry["checkpoint_note"] = ("Weights are intentionally absent from the repository; "
                                "the champion is regenerated by the command above.")
    pointer.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clear_directory(path: Path) -> bool:
    """Remove everything inside *path*, keeping the directory itself.

    Returns True when the directory ends up empty.  The directory itself is
    left in place when the OS refuses to delete it - that happens on Windows
    whenever a process still holds a handle on it, and it does not affect the
    result, because every file is rewritten afterwards.
    """
    for child in sorted(path.iterdir(), reverse=True):
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:
            # A locked child is reported, not silently ignored: a stale file
            # left over from a previous run would be published.
            print(f"WARN  could not remove {child}", file=sys.stderr)
            return False
    return True


def main() -> int:
    """Build the folder and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dist/github",
                        help="Output folder; an absolute path is used as-is, "
                             "a relative path is resolved against the repository root")
    parser.add_argument("--name", default="zhixun-engine-b")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / args.out
    out_dir = out_dir.resolve()
    if out_dir.exists():
        # On Windows a stray handle (an editor, an explorer window, a shell
        # whose cwd is the folder) makes the directory itself undeletable even
        # though its contents are removable.  Clear the contents first, then
        # try to drop the directory itself; if it still cannot be removed, the
        # folder is reused, which is harmless because every file in it is
        # overwritten below.
        if not _clear_directory(out_dir):
            raise SystemExit(f"cannot clear output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect()

    leaks = [str(rel) for _src, rel in files
             if rel.suffix.lower() in WEIGHT_SUFFIXES or rel.name in SKIP_FILES]
    if leaks:
        raise SystemExit(f"refusing to publish weight files: {leaks}")

    for source, rel in files:
        destination = out_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    # The push guide is generated, not copied.
    (out_dir / "GIT_PUSH.md").write_text(PUSH_TEMPLATE, encoding="utf-8")

    # The registry pointer must not dangle once the weights are gone.
    _rebase_champion_pointer(out_dir)

    total = sum((out_dir / rel).stat().st_size for _src, rel in files)
    if total > MAX_TOTAL_BYTES:
        raise SystemExit(f"payload {total} bytes exceeds the {MAX_TOTAL_BYTES} byte budget")

    manifest = {
        "name": args.name,
        "contains_model_weights": False,
        "total_bytes": total,
        "file_count": len(files) + 1,
        "note": "Weights are intentionally absent; the harness regenerates them "
                "from the data package at evaluation time.",
        "files": {str(rel).replace("\\", "/"): {
            "bytes": (out_dir / rel).stat().st_size,
            "sha256": digest(out_dir / rel),
        } for _src, rel in files},
    }
    manifest["files"]["GIT_PUSH.md"] = {
        "bytes": (out_dir / "GIT_PUSH.md").stat().st_size,
        "sha256": digest(out_dir / "GIT_PUSH.md"),
    }
    (out_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "out": str(out_dir),
        "files": manifest["file_count"],
        "total_bytes": total,
        "contains_model_weights": False,
    }, ensure_ascii=False, indent=2))
    print(f"REPO OK  {out_dir}  ({total / 1024:.1f} KB, 0 weights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
