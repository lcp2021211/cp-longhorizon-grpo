# 中文文档

项目的中文主文档已经统一到根目录 [README.md](README.md)，以避免两份运行命令和实验口径发生漂移。

- [项目概览与快速开始](README.md)
- [完整训练、消融与评测指南](agentic-grpo-longhorizon/docs/training_and_ablation_guide.md)
- [SALT + ProGPO + LATA 实现说明](agentic-grpo-longhorizon/docs/salt_progpo_lata_tau2.md)

当前主路径使用 current tau2-bench airline official train/test split，并提供 SALT × ProGPO × LATA 的 `2^3` 全因子消融。历史 50-task `tau-bench` 结果仅作项目演进记录，不能与新的 tau2 实验直接横比。
