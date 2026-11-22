# 基岩智序 · 千问 SFT 微调交接说明

日期：2026-11-22

## 1. 本次取得什么
- 基础模型：Qwen/Qwen3.5-4B
- 方法：PyTRIO LoRA/SFT
- 训练案例：24；保留案例：6
- 保留样本标签匹配率：基础模型 66.7%，微调后 100.0%（仅6条）
- 云端 Sampler/LoRA 权重完整路径：

  trio://run_rru8f96z583a/sampler_weights/jyzx-review-sft-20261122135410

**注意：`trio://` 是 PyTRIO 平台的权重定位符，不是 Windows 本地文件；不需要发送自己的 API Key。**

## 2. 交接文件
- `sft_repo_report.json`：本次结果，含微调前后逐例回答、Loss 和权重路径。
- `run_repo_sft.py`：训练/检查/评估脚本（存档复现；没有自动运行）。
- `train_repo.jsonl` / `eval_repo.jsonl`：本次24/6条训练与保留案例。
- `manifest.json`：来源、训练/保留划分与局限说明。

## 3. 请集成负责人做什么
1. 在比赛平台选择与权重匹配的基础模型（Qwen/Qwen3.5-4B）和「持久化 Sampler」方式；填入上述完整 `trio://` 权重路径；确认平台支持该格式以及此权重具有可访问权限。
2. 用正式 Harness 的 `_model_brief()` 原始输入和 `rnd.model(...)` 调用，单独测试基础模型 vs 微调模型的输出。**正式提交镜像里，模型是实验完成之后进行结果审核，并不控制该轮的假设选择/晋级决策。**
3. 评估回答质量、输出格式、调用成功率和延迟；确认不会影响当前可运行 Docker 镜像。
4. 不要把这次 6/6 当作竞赛分数或者独立泛化证明。

## 4. 重要限制
- 30条原始案例来自同一次公开数据离线演练；训练和保留案例存在关联。
- 监督标签（尤其 verdict 和 next_test）由模板从真实指标和账本构造，**不是队友人工验证的真值**。
- 模型回答有科研逻辑瑕疵，人工审核后才适合作为正式训练或答辩性能依据。
- 使用 sampler 并不自动改善工业模型 A/B/C 的指标，需对比实际比赛端到端行为。
- 不要分享账号密码、API Key、私人 Token，也不要把它们写进代码或仓库。

## 5. 源码版本
GitHub: https://github.com/GY-phd/ai-
main commit: 26bbeed2f08c5dcfb3de2cde87420f848b9db439
关键提交代码：contest_baselines/layers/controller/round.py
实验账本：runs/harness-offline-10/ledger.jsonl
