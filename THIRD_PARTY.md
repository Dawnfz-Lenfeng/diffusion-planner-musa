# 第三方依赖与 fork 基线（锁定 commit，不入库的放 third_party/）

| 组件 | 版本/commit | 说明 |
|---|---|---|
| Diffusion-Planner（fork base） | `a3a621f0b724c5fa6447f7a2fbaf9e0387bd35df`（main, 2026-03-10） | 已导入本仓库根目录；remote `upstream` 可对照 |
| nuplan-devkit | `master` 浅克隆于 2026-11-20（commit 见 `git -C third_party/nuplan-devkit log -1`） | third_party/，不改动，`pip install -e --no-deps` |
| 官方权重 | HF `ZhengYinan2001/Diffusion-Planner`（args.json + model.pth，2025-03-18 版） | checkpoints/；model.pth 不入库 |
| nuPlan maps | nuplan-maps-v1.0（zip 971,557,640 B） | data/，不入库 |
| nuPlan mini | nuplan-v1.1 mini（zip 8,550,103,030 B） | data/，不入库 |

刷新 upstream：`git fetch upstream && git log upstream/main --oneline -5`
