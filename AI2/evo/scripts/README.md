# scripts/ — 工具脚本

本目录放**可复现的实验与合规工具**。所有脚本都可以直接跑, 不需要额外配置。

| 脚本 | 用途 | 何时用 |
|---|---|---|
| `build_submission.py` | 构建赛题提交包 (白名单收集 + 权重排除 + SHA-256 清单 + 体积校验) | **提交前必跑** |
| `compare_official_baseline.py` | 与官方 `checkpoint.npz` 直接对照, 算出"比官方基线好多少 %" | 答辩前必跑 |
| `bench_rul_models.py` | RUL 候选模型的横向对比 (MLP 容量 / 树模型 / 集成 / 目标缩放) | 换基线模型前必跑 |
| `validate_scenarios.py` | 全场景验证 —— 覆盖赛题三条基础门槛 + 鲁棒性场景 | 交付前必跑 |
| `audit_rul.py` | RUL 闭环的严格性审查 —— 检查是否存在评估泄漏 | 发现指标异常时 |
| `compare_rul.py` | 赛题 B 对照实验 —— 定量证明自进化闭环的价值 | 写设计论证时 |
| `compare_systems.py` | 对照实验 —— 定量证明"完整系统 > 静态基线" | 写设计论证时 |
| `sweep_rul.py` | RUL 参数扫描 —— 影子集大小 / 探索率的最优组合 | 调参时 |
| `inspect_official.py` | 分析官方数据包: 轴承划分、RUL 分布、基线性能 | 理解数据时 |
| `inspect_rul_explore.py` | 检查 RUL 场景下策略层的探索是否真的有效 | 怀疑探索空转时 |
| `inspect_shadow.py` | 查看影子评估的拒绝原因分布 | 怀疑"从不上线"时 |

## 约定

- **脚本只读仓库内数据**, 不修改 `src/` 下的代码 —— 修复逻辑一律回到 `src/` 里改,
  否则评审看到的代码与实验结果对不上。
- 所有脚本用 `sys.path.insert(0, ROOT / "src")` 引入包, 可直接 `python scripts/xxx.py` 运行。
- 实验产物一律落在 `artifacts/` 下 (已在 `.gitignore` 中), 不污染仓库。
- 脚本内的数据路径用 `--data-root` 等参数传入, 默认值指向本机赛题包位置。

## 注意

`build_submission.py` 收集的是 **仓库源码本身** (不是 `dist/` 里的副本),
所以改完代码直接重新构建即可, 无需先同步目录。
