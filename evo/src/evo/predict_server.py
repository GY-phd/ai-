"""官方评测协议适配器 —— 让 Harness 直接对接赛题平台。

**协议来源**: 赛题包 `baselines/predict.py` 与 `baselines/data.py`。

请求格式 (stdin, 逐行 JSON):
    {"task": "regression", "inputs": {...嵌套数值...}}

响应格式 (stdout, 逐行 JSON):
    * 回归:     {"outputs": {"value": 非负数}}
    * 分类:     {"outputs": [{"label": "Ethanol", "score": 0.87}, ...]}

关键约束 (来自官方源码):
  * 请求只能含 `task` 与 `inputs` 两个键, 多余键直接报错
  * 预测的**特征 schema 必须与检查点一致** (paths 完全相同), 否则报错
  * 输出必须非负 (RUL 截断至非负值)
  * JSON 不允许 NaN/Inf (`allow_nan=False`)
  * 不得把轴承编号/快照序号/总快照数/目标值拼进输入

本模块提供:
  * `PredictServer` — 常驻服务, 从 stdin 读请求、输出响应
  * `predict_file` — 离线批量预测 (写预测文件)
  * `validate_task_output_format` — 输出格式校验

用途:
  * 评测平台拉起我们的镜像后, 由 `python -m evo serve` 接管 stdin/stdout
  * 本地可用 `python -m evo predict --input req.jsonl --output pred.jsonl` 校验格式

**重要**: 特征 schema 一致性检查必须严格 —— 官方 loader 会比对 paths,
不一致直接抛错。我们这里在**训练时**就固定 paths, 推理时逐条比对,
避免"静默错位"这种最难查的 bug。
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import numpy as np

from .data import _flatten_jsonl_inputs


# =====================================================================
# 工具
# =====================================================================

def _reject_nonfinite(_: str) -> float:
    raise ValueError("nonfinite JSON number")


def parse_json_strict(raw: str) -> Any:
    """严格 JSON 解析 —— 与官方 json_value 语义对齐 (拒绝 NaN/Inf/重复键)。"""
    def unique_object(pairs):
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    return json.loads(
        raw,
        object_pairs_hook=unique_object,
        parse_constant=_reject_nonfinite,
    )


def format_response(payload: Dict[str, Any]) -> str:
    """序列化响应 —— 官方用 allow_nan=False, 这里同样保证。"""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)


# =====================================================================
# 预测服务
# =====================================================================

@dataclass
class PredictServer:
    """官方协议的常驻预测服务。

    用法:
        server = PredictServer(model_bundle, task="regression", classes=None)
        server.serve(sys.stdin, sys.stdout)
    """

    model: Any                      # ModelBundle
    task: str = "regression"
    classes: Optional[List[str]] = None
    expected_paths: Optional[List[str]] = None
    # 统计 (便于评估资源使用)
    n_requests: int = 0
    n_errors: int = 0
    total_predict_seconds: float = 0.0

    def __post_init__(self) -> None:
        # 特征 schema: 优先用模型训练时记录的; 否则从模型 bundle 取
        if self.expected_paths is None:
            paths = getattr(self.model, "feature_paths", None)
            if paths:
                self.expected_paths = list(paths)

    # -- 单条请求 ------------------------------------------------------
    def handle(self, request: Any) -> Dict[str, Any]:
        """处理单个请求, 返回响应 payload。"""
        if not isinstance(request, dict) or set(request) - {"task", "inputs"} or "inputs" not in request:
            raise ValueError("prediction request requires only task and inputs")
        req_task = request.get("task", self.task)
        if req_task != self.task:
            raise ValueError("prediction task mismatch")

        paths, values = _flatten_jsonl_inputs(request["inputs"])
        if self.expected_paths is not None and paths != list(self.expected_paths):
            raise ValueError("prediction feature schema differs from checkpoint")

        import time

        t0 = time.perf_counter()
        pred = self.model.predict(np.asarray([values], dtype=np.float64))[0]
        self.total_predict_seconds += time.perf_counter() - t0

        if self.task == "regression":
            # 官方: 预测还原为分钟/循环数并**截断至非负值**
            return {"outputs": {"value": max(0.0, float(pred))}}
        # 分类: 输出每类的标签与分数
        arr = np.asarray(pred, dtype=np.float64)
        if self.classes is None:
            classes = [str(i) for i in range(len(arr))]
        else:
            classes = list(self.classes)
        if len(arr) != len(classes):
            raise ValueError("class count mismatch between model output and config")
        return {
            "outputs": [
                {"label": label, "score": float(score)}
                for label, score in zip(classes, arr)
            ]
        }

    # -- 常驻循环 ------------------------------------------------------
    def serve(self, stdin: TextIO, stdout: TextIO) -> int:
        """从 stdin 逐行读请求, 向 stdout 逐行写响应。

        与官方 predict.py 完全一致的交互:
          * 每行一个 JSON 请求
          * 每行一个 JSON 响应, flush=True
          * 单条出错**不终止服务** —— 写错误行并继续, 避免一条脏数据
            拖垮整个评测
        """
        for line in stdin:
            if not line.strip():
                continue
            self.n_requests += 1
            try:
                request = parse_json_strict(line)
                response = self.handle(request)
            except Exception as exc:
                self.n_errors += 1
                # 错误也用 JSON 输出, 让评测方能识别
                stdout.write(
                    format_response({"error": f"{type(exc).__name__}: {exc}"}) + "\n"
                )
                stdout.flush()
                continue
            stdout.write(format_response(response) + "\n")
            stdout.flush()
        return 0


# =====================================================================
# 离线批量预测
# =====================================================================

def predict_file(
    model: Any,
    task: str,
    input_path: str | Path,
    output_path: str | Path,
    classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """离线批量预测: 读 JSONL 请求, 写 JSONL 响应。

    用于本地校验输出格式是否符合官方协议, 不依赖平台。
    """
    server = PredictServer(model=model, task=task, classes=classes)
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_err = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            server.n_requests += 1
            try:
                request = parse_json_strict(line)
                response = server.handle(request)
                n_ok += 1
            except Exception as exc:
                server.n_errors += 1
                n_err += 1
                response = {"error": f"{type(exc).__name__}: {exc}"}
            fout.write(format_response(response) + "\n")

    return {
        "n_requests": server.n_requests,
        "n_ok": n_ok,
        "n_errors": n_err,
        "output": str(out_path),
        "total_predict_seconds": round(server.total_predict_seconds, 4),
    }


def validate_task_output_format(task: str, outputs: Any) -> None:
    """校验响应格式符合官方协议 (供测试使用)。

    官方要求:
      * 回归: outputs 是 {"value": 非负数}
      * 分类: outputs 是 [{"label": str, "score": float}, ...]
    """
    if task == "regression":
        if not isinstance(outputs, dict) or "value" not in outputs:
            raise ValueError('regression output must be {"value": number}')
        v = outputs["value"]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("regression value must be a number")
        if not math.isfinite(float(v)):
            raise ValueError("regression value must be finite")
        if float(v) < 0:
            raise ValueError("regression value must be nonnegative")
        return

    if not isinstance(outputs, list) or not outputs:
        raise ValueError("classification output must be a nonempty list")
    for item in outputs:
        if not isinstance(item, dict) or "label" not in item or "score" not in item:
            raise ValueError('classification item must be {"label":..., "score":...}')
        if not isinstance(item["label"], str):
            raise ValueError("classification label must be a string")
        if isinstance(item["score"], bool) or not isinstance(item["score"], (int, float)):
            raise ValueError("classification score must be a number")
        if not math.isfinite(float(item["score"])):
            raise ValueError("classification score must be finite")
