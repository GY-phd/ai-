"""提交包构建 — 赛题合规打包。

赛题硬性约束:
  * 提交包大小 ≤ 2G
  * **严禁将模型权重打包进提交包**
  * 需包含: 系统镜像、源代码与依赖、Harness 与模型配置、
            Swan Lab 记录、设计论证、运行说明、开源合规展示
  * 评测环境: 四核八G Linux, **无网络**

本脚本做的事:
  1. 按白名单收集源码与配置 (而非整目录拷贝) —— 从根上杜绝误带权重。
  2. 显式排除权重文件 (*.pkl / *.joblib / *.pt / *.pth / *.onnx / *.bin /
     *.safetensors / *.h5 / *.ckpt) 与产物目录 artifacts/models。
  3. 产出 `submission_manifest.json` —— 供评委与自查核对内容清单。
  4. 校验包大小 ≤ 2G, 超限直接失败。

用法:
    python scripts/build_submission.py --out dist/submission
    python scripts/build_submission.py --out dist/submission --check-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]

# ---- 权重类文件后缀 (赛题明令禁止打包) ----
WEIGHT_SUFFIXES = {
    ".pkl", ".pickle", ".joblib", ".pt", ".pth", ".onnx", ".bin",
    ".safetensors", ".h5", ".hdf5", ".ckpt", ".npz", ".npy", ".pmml",
}

# ---- 明确排除的目录 ----
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".idea", ".vscode",
    "artifacts", "dist", "data", "outputs", "logs", ".ipynb_checkpoints",
}

# ---- 需要随包提交的文件 (赛题材料清单) ----
REQUIRED_FILES = [
    "README.md",
    "pyproject.toml",
    "configs/default.yaml",
]

# ---- 建议随包提交的材料 (缺失只警告, 不阻断) ----
RECOMMENDED_FILES = [
    "docs/设计论证.md",
    "docs/系统运行说明.md",
    "docs/数据适配说明.md",
    "docs/实验计划.md",
    "docs/训练凭证.md",
]


def _is_weight(path: Path) -> bool:
    return path.suffix.lower() in WEIGHT_SUFFIXES


def _collect_files() -> List[Path]:
    """按白名单收集要打包的文件。

    只收: src/ 下代码、configs/ 配置、scripts/ 脚本、tests/ 测试、
    docs/ 文档、根目录的说明与依赖文件。
    """
    out: List[Path] = []
    include_dirs = ["src", "configs", "scripts", "tests", "docs"]
    include_globs = ["*.md", "*.toml", "*.txt", "*.yaml", "*.yml", "*.cfg", "*.lock"]

    for d in include_dirs:
        base = ROOT / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            if any(part in EXCLUDE_DIRS for part in p.parts):
                continue
            if _is_weight(p):
                continue
            out.append(p)

    for pattern in include_globs:
        for p in ROOT.glob(pattern):
            if p.is_file() and not _is_weight(p):
                out.append(p)

    # 去重 + 相对路径排序
    seen: Dict[str, Path] = {}
    for p in out:
        rel = p.relative_to(ROOT).as_posix()
        seen[rel] = p
    return [seen[k] for k in sorted(seen)]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(out_dir: Path, check_only: bool = False) -> int:
    files = _collect_files()

    # ---- 权重泄漏检查 (硬性) ----
    leaked = [p.relative_to(ROOT).as_posix() for p in files if _is_weight(p)]
    if leaked:
        print("严重错误: 收集到权重文件, 已中止:", leaked, file=sys.stderr)
        return 2

    # ---- 必需文件检查 ----
    missing_req = [f for f in REQUIRED_FILES if not (ROOT / f).exists()]
    if missing_req:
        print(f"警告: 缺少赛题要求的文件: {missing_req}", file=sys.stderr)
    missing_rec = [f for f in RECOMMENDED_FILES if not (ROOT / f).exists()]
    if missing_rec:
        print(f"提示: 缺少建议材料: {missing_rec}", file=sys.stderr)

    total = sum(p.stat().st_size for p in files)
    total_mb = total / (1 << 20)
    limit_gb = 2.0

    manifest = {
        "name": "zhixun-engine-submission",
        "files": [],
        "total_bytes": total,
        "total_mb": round(total_mb, 3),
        "size_limit_gb": limit_gb,
        "size_ok": total_mb < limit_gb * 1024,
        "weight_files_included": [],
        "excluded_dir_names": sorted(EXCLUDE_DIRS),
        "weight_suffix_blocked": sorted(WEIGHT_SUFFIXES),
        "missing_required": missing_req,
        "missing_recommended": missing_rec,
        "runtime_env": {
            "cpu_cores": 4,
            "memory_gb": 8,
            "gpu": False,
            "network": False,
            "entrypoint": "python -m evo run --config configs/default.yaml",
            "python": ">=3.10",
        },
    }
    for p in files:
        rel = p.relative_to(ROOT).as_posix()
        manifest["files"].append(
            {"path": rel, "bytes": p.stat().st_size, "sha256": _sha256(p)}
        )

    if check_only:
        print(json.dumps({k: v for k, v in manifest.items() if k != "files"},
                         ensure_ascii=False, indent=2))
        return 0 if manifest["size_ok"] else 2

    # ---- 实际拷贝 ----
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in files:
        rel = p.relative_to(ROOT)
        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)

    manifest_path = out_dir / "submission_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"提交包已构建: {out_dir}")
    print(f"  文件数: {len(files)}")
    print(f"  总体积: {total_mb:.2f} MB (上限 {limit_gb * 1024:.0f} MB)")
    print(f"  权重文件: 0 (已强制排除)")
    print(f"  清单: {manifest_path}")
    if not manifest["size_ok"]:
        print("错误: 提交包超出 2G 上限", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="构建赛题提交包")
    ap.add_argument("--out", default="dist/submission", help="输出目录")
    ap.add_argument("--check-only", action="store_true", help="只检查不拷贝")
    args = ap.parse_args()
    return build(Path(args.out), check_only=args.check_only)


if __name__ == "__main__":
    raise SystemExit(main())
