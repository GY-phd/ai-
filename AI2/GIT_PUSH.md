# 上传到 GitHub

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
docker run --rm --network none \
  -v /path/to/official-data:/data:ro \
  -v /path/to/output:/out \
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
