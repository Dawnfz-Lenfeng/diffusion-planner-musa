# Diffusion Planner × nuPlan mini — MUSA 基线交付

For MUSA users（摩尔线程 GPU，已有 torch_musa 工具链的环境）：直接装在现有环境之上，无需专属镜像：

```bash
pip install -r docker/requirements-lock.txt            # 213 个精确锁定的依赖
git clone https://github.com/motional/nuplan-devkit third_party/nuplan-devkit \
  && git -C third_party/nuplan-devkit checkout e9241677997dd86bfc0bcd44817ab04fe631405b \
  && pip install --no-deps -e third_party/nuplan-devkit
pip install --no-deps -e .
export NUPLAN_DATA_ROOT=<data> NUPLAN_MAPS_ROOT=<maps> NUPLAN_EXP_ROOT=<exp>
bash scripts/preprocess_mini.sh && bash torch_run_musa.sh 8 --train_epochs 500
```

详细步骤与已知坑见 `docs/REPRODUCE.md`；可选 Docker 配方见 `docker/`。

Diffusion Planner（ICLR 2025 Oral）在 nuPlan mini 上基于 **MT MUSA GPU 栈**（torch_musa + MCCL）的
完整基线交付：数据预处理 → 8 卡训练（500 epochs）→ 闭环仿真评测 → 聚合打分，含
官方权重与自训权重的同口径对比。

## 交付内容

| 目录/文件 | 说明 |
|---|---|
| `diffusion_planner/` | 基线模型代码（fork 自官方 release，适配已内联，见下「相对上游的改动」） |
| `data_process.py` | 数据预处理入口（`--log_list` / `--num_proc`，支持任意 split 与并行） |
| `train_predictor.py` | 训练入口（含 `--device musa`、faulthandler 等适配） |
| `torch_run_musa.sh` | MUSA 多卡训练启动器（MCCL 环境约定） |
| `scripts/` | 数据下载、预处理、评测两阶段流水线、分数读取 |
| `configs/official_args.json` | 官方 checkpoint 配套配置（评测必需） |
| `docs/REPRODUCE.md` | 环境搭建与全流程复现指南 |
| `docs/musa-port.md` | MUSA 移植说明（改动清单 + 验证阶梯） |
| `docs/debug-log.md` | 踩坑记录（21 项，含解法） |
| `docker/` | **MUSA 版镜像配方（主交付）** + CUDA 参照镜像；数据/权重运行时挂载 |
| `THIRD_PARTY.md` | 上游/依赖锁定信息 |

**数据与权重不随仓库分发**：nuPlan 数据需按官方条款自行下载（见 REPRODUCE.md）；
官方权重从 HuggingFace `ZhengYinan2001/Diffusion-Planner` 获取（约 45MB）。

## 评测与打分

评测为两阶段闭环流程（`run_simulation.py` 仿真 → `run_metric.py` 计算指标并聚合），
`scripts/sim_eval.sh` 跑完后用 `scripts/read_scores.py` 读取 `final_score` 与分项指标。
**本仓库不发布任何性能 / 分数数据**，请在自己的环境上运行得出。

## 相对上游的改动（均已内联，无 patch 文件）

| 文件 | 改动 |
|---|---|
| `diffusion_planner/utils/accelerator.py` | 新增：cuda/musa/cpu 三态兼容层（设备/同步/分布式后端 nccl↔mccl↔gloo） |
| `diffusion_planner/utils/ddp.py` | torch.cuda 触点走兼容层；尊重 torchrun 注入的 rendezvous |
| `diffusion_planner/train_epoch.py`、`train_predictor.py`、`planner/planner.py` | 设备归一化 + musa 支持 + `--guidance_fn` 缺省（官方 args.json 早于该功能） |
| `diffusion_planner/data_process/data_processor.py` | 场景提取 joblib 并行 + 失败隔离 + 断点续跑 + 原子写 |
| `data_process.py` | `--log_list`（任意 split 的 log 名单）+ `--num_proc` |

上游：ZhengYinan-AIR/Diffusion-Planner @ a3a621f（详见 THIRD_PARTY.md）。

## 环境交付形式

- **主路径：源码安装到已有 MUSA 环境**——`docker/requirements-lock.txt`（213 包精确版本）+
  `docs/REPRODUCE.md` 逐条命令。MUSA 工具链（torch_musa / mcc）
  由你的环境提供；本仓库不绑定、也不发布基础镜像。
- **可选：Docker 配方**（`docker/`）——`Dockerfile` 的 `FROM` 填你侧的 MUSA 基础镜像 tag 后构建；
  `Dockerfile.cuda` 为 CUDA 参照（公开 pytorch 基础镜像，任意 x86 docker 机器可构建）。
  镜像内安装的依赖与主路径同为 `docker/requirements-lock.txt`，行为一致。

## 快速开始

```bash
# 0. 数据 + 官方权重（约 9.6GB，断点续传、带字节数校验）
bash scripts/download_data.sh "$DATA_ROOT"

# 1. 环境（详见 docs/REPRODUCE.md：py3.10 + 钉版依赖清单 scripts/requirements-nuplan-x86.txt）
#    MUSA 机器使用镜像自带 torch/torch_musa，勿用 pip 安装 torch

# 2. 预处理（生成训练 .npz）
bash scripts/preprocess_mini.sh

# 3. 训练（8 卡 MUSA）
bash torch_run_musa.sh 8 --batch_size 2048 --train_epochs 500

# 4. 评测（两阶段：闭环仿真 → 指标计算聚合）
FILTER=mini_eval DEVICE=musa CKPT=<checkpoint.pth> ARGS=<args.json> bash scripts/sim_eval.sh
python scripts/read_scores.py <metric_run_dir>
```
