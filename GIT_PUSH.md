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

赛题要求提交「模型配置 + Harness 的系统镜像」。**提交镜像由仓库根目录的
`Dockerfile` 定义**：

```bash
docker build -t zhixun-engine-b .
```

镜像特征：

* **入口唯一**：SDK 的固定入口 `python3 /app/harness.py describe|train|predict`，
  落到 `contest_submission.py` → `contest_baselines/`；镜像里**没有**第二套
  候选入口，也没有同名包遮蔽入口；
* **只依赖 `numpy`**（**不装 swanlab**：评测环境离线，本地已按 SwanLab 形状写指标流，
  云端客户端只会给一次不该失败的运行增加失败模式）；
* **A / B / C 同一路径**：`classification` 与 `regression` 走同一套
  `train` / `predict`，训练每轮含一次真实 `rnd.model` 调用；
* 构建时会跑 `tools/verify_repo.py`，**镜像里不可能混进权重文件**；
* 以非 root 用户（uid 10001）运行。

> 复现本仓库 README 里的闭环数字，用的是**另一个**镜像
> `Dockerfile.research`（`docker build -f Dockerfile.research`）——
> 那是研究运行时，不是提交镜像。

更详细的运行手册见 `docs/RUNBOOK.md`。

## 5. 说明

- **权重为什么不在这里**：赛题要求提交的是"可复现的代码 + 由代码产出的模型"。
  仓库里的 `evidence/` 只保留账本、运行报告与指标流（纯文本），
  `*.npz` 检查点由训练代码在评测时重新生成。
- **证据链**：`evidence/ledger.jsonl` 是哈希链账本，CI 会重新校验；
  任何一条记录被改动，CI 就会变红。
- **`harness/hypothesis.py`** 是融合进来的四段式证据链
  （读历史 -> 提假设 -> 做实验 -> 做决策），对应赛题门槛 ②③。
