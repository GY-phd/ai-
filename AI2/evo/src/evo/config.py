"""配置加载与路径解析。"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class Config:
    """轻量配置容器，支持点号访问与覆盖。"""

    def __init__(self, data: Dict[str, Any], root: Path | None = None):
        self._data = data
        self.root = root or Path.cwd()

    # -- 访问 ------------------------------------------------------------
    def get(self, path: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set(self, path: str, value: Any) -> None:
        parts = path.split(".")
        node = self._data
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def section(self, path: str) -> Dict[str, Any]:
        val = self.get(path, {})
        return copy.deepcopy(val) if isinstance(val, dict) else {}

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self._data)

    def update(self, overrides: Dict[str, Any]) -> "Config":
        """用扁平化的点号 key 覆盖配置。"""
        for key, value in overrides.items():
            self.set(key, value)
        return self

    # -- 路径 ------------------------------------------------------------
    def path(self, key: str) -> Path:
        """解析配置里的相对路径为绝对路径。"""
        raw = Path(str(self.get(key, "")))
        return raw if raw.is_absolute() else (self.root / raw)

    def ensure_dirs(self) -> None:
        for key in (
            "paths.artifacts_root",
            "paths.audit_log",
            "paths.shadow_root",
            "experiment.local_dir",
        ):
            self.path(key).parent.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # 环境变量覆盖（评测环境离线注入）
    for env_key, cfg_key in (
        ("PYTRIO_BASE_URL", "llm.base_url"),
        ("PYTRIO_API_KEY", "llm.api_key"),
        ("PYTRIO_MODEL", "llm.model"),
    ):
        val = os.environ.get(env_key)
        if val:
            data.setdefault("llm", {})[cfg_key.split(".")[-1]] = val

    cfg = Config(data, root=root or cfg_path.parent.parent)
    cfg.ensure_dirs()
    return cfg
